#!/usr/bin/env python3
"""
OCI Image Unpacking Script for Singularity/Apptainer

Downloads and unpacks OCI container images using Singularity, with intelligent
caching based on manifest hash comparison to avoid unnecessary downloads.

Compatible with Python 3.6.8
"""

import argparse
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path
from subprocess import PIPE

try:
    import docker
except ImportError:
    print("Error: docker library is required. Install with: pip install docker", file=sys.stderr)
    sys.exit(1)


class ImageUnpackerError(Exception):
    """Base exception for image unpacker errors."""
    pass


class ManifestFetchError(ImageUnpackerError):
    """Error fetching image manifest."""
    pass


class ImageDownloadError(ImageUnpackerError):
    """Error downloading/building image."""
    pass


class ConfigError(ImageUnpackerError):
    """Configuration or remote file error."""
    pass


class ImageUnpacker:
    """Handles OCI image downloading and unpacking with Singularity."""
    
    def __init__(self, image_dir, remote_url):
        self.image_dir = Path(image_dir)
        self.remote_url = remote_url
        self.images = None
        self.logger = self._setup_logging()
        
    def _setup_logging(self):
        """Set up logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        return logging.getLogger(__name__)
    
    def load_images_from_remote(self, remote_url):
        """Load image list from remote URL."""
        try:
            self.logger.info("Fetching image list from: {}".format(remote_url))
            
            with urllib.request.urlopen(remote_url, timeout=30) as response:
                content = response.read().decode('utf-8')
            
            # Parse lines, strip whitespace, and filter out empty lines and comments
            images = []
            for line_num, line in enumerate(content.splitlines(), 1):
                line = line.strip()
                if line and not line.startswith('#'):
                    images.append(line)
            
            if not images:
                raise ConfigError("No images found in remote file")
            
            self.logger.info("Loaded {} images from remote".format(len(images)))
            return images
            
        except Exception as e:
            raise ConfigError("Failed to fetch image list from remote: {}".format(e))
    
    def parse_image_name(self, image_name):
        """
        Parse image name into components.
        
        Examples:
        - nginx:latest -> docker.io/library/nginx:latest
        - busybox -> docker.io/library/busybox:latest
        - docker.io/nginx:latest -> docker.io/library/nginx:latest
        - quay.io/biocontainers/blast:2.9.0 -> quay.io/biocontainers/blast:2.9.0
        """
        if '://' in image_name:
            # Remove protocol if present
            image_name = image_name.split('://', 1)[1]
        
        # Split by '/' to get parts
        parts = image_name.split('/')
        
        # Determine registry, project, name, and tag
        if len(parts) == 1:
            # Just image name, e.g., "nginx:latest"
            registry = "docker.io"
            project = "library"
            name_tag = parts[0]
        elif len(parts) == 2:
            if parts[0] == "docker.io":
                # docker.io/nginx:latest -> library namespace (Docker Hub special case)
                registry = "docker.io"
                project = "library"
                name_tag = parts[1]
            else:
                # Project with image, e.g., "opensciencegrid/osgvo:latest" or "microsoft.com/windows:latest"
                registry = "docker.io"
                project = parts[0]
                name_tag = parts[1]
        elif len(parts) > 2:
            # Full format, e.g., "docker.io/opensciencegrid/osgvo:latest"
            # or more than 3 parts: "gitlab-registry.nrp-nautilus.io/prp/perfsonar/meshconfig:latest"
            registry = parts[0]
            project = parts[1]
            name_tag = '/'.join(parts[2:])
        else:
            raise ValueError("Invalid image name format: {}".format(image_name))
        
        # Split name and tag
        if ':' in name_tag:
            name, tag = name_tag.rsplit(':', 1)
        else:
            name, tag = name_tag, 'latest'
        return {
            'registry': registry,
            'project': project,
            'name': name,
            'tag': tag
        }
    
    def get_image_manifest_hash(self, image_name):
        """
        Fetch the manifest hash for an image using Docker SDK.
        
        Returns the sha256 hash of the image manifest.
        """
        try:
            client = docker.from_env()
            
            # Get registry data (this fetches manifest without pulling image)
            registry_data = client.images.get_registry_data(image_name)
            
            # Extract the digest (manifest hash)
            if hasattr(registry_data, 'id') and registry_data.id:
                return registry_data.id
            elif hasattr(registry_data, 'attrs') and 'Descriptor' in registry_data.attrs:
                return registry_data.attrs['Descriptor']['digest']
            else:
                raise ManifestFetchError("Could not extract manifest hash from registry data")
                
        except docker.errors.ImageNotFound:
            raise ManifestFetchError("Image not found: {}".format(image_name))
        except docker.errors.APIError as e:
            raise ManifestFetchError("Docker API error for {}: {}".format(image_name, e))
        except Exception as e:
            raise ManifestFetchError("Failed to fetch manifest for {}: {}".format(image_name, e))
    
    def get_hash_directory(self, image_hash):
        """
        Convert image hash to directory path.
        
        sha256:abc123... -> .images/sha256:ab/c123.../
        """
        if not image_hash.startswith('sha256:'):
            raise ValueError("Expected sha256 hash, got: {}".format(image_hash))
        
        hash_part = image_hash[7:]  # Remove 'sha256:' prefix
        hash_prefix = hash_part[:2]
        hash_suffix = hash_part[2:]
        
        return self.image_dir / '.images' / 'sha256:{}'.format(hash_prefix) / hash_suffix
    
    def get_symlink_path(self, image_name):
        """Get the symlink path for an image."""
        parsed = self.parse_image_name(image_name)
        
        return self.image_dir / parsed['registry'] / parsed['project'] / "{}:{}".format(parsed['name'], parsed['tag'])
    
    def get_current_hash_from_symlink(self, symlink_path):
        """Extract current hash from existing symlink if it exists."""
        if not symlink_path.exists() or not symlink_path.is_symlink():
            return None
        
        try:
            target = symlink_path.resolve()
            # Extract hash from path like .images/sha256:ab/c123.../
            target_str = str(target)
            match = re.search(r'\.images/sha256:([a-f0-9]{2})/([a-f0-9]+)', target_str)
            if match:
                hash_prefix = match.group(1)
                hash_suffix = match.group(2)
                return "sha256:{}{}".format(hash_prefix, hash_suffix)
        except (OSError, ValueError):
            pass
        
        return None
    
    def download_image(self, image_name, hash_dir):
        """Download and unpack image using Singularity."""
        self.logger.info("Downloading image: {}".format(image_name))
        
        # Ensure parent directory exists
        hash_dir.parent.mkdir(parents=True, exist_ok=True)
        
        
        # Use temporary directory for atomic operation
        with tempfile.TemporaryDirectory(dir=str(hash_dir.parent)) as temp_dir:
            temp_path = Path(temp_dir) / 'sandbox'
            
            try:
                # Build image with Singularity (run in temp directory to avoid polluting cwd)
                cmd = [
                    'singularity', '--silent', 'build',
                    '--tmpdir',
                    '/var/tmp',
                    '--disable-cache',
                    '--force',
                    '--fix-perms',
                    '--sandbox',
                    str(temp_path),
                    'docker://{}'.format(image_name)
                ]
                
                self.logger.info("Running: {}".format(' '.join(cmd)))
             
                result = subprocess.run(
                    cmd,
                    stdout=PIPE,
                    stderr=PIPE,
                    check=True,
                    timeout=1800,  # 30 minute timeout
                    cwd=str(temp_dir)  # Run in temporary directory
                )   
               
                # Move from temp location to final location atomically
                if hash_dir.exists():
                   shutil.rmtree(str(hash_dir))
                
                shutil.move(str(temp_path), str(hash_dir))
                
                self.logger.info("Successfully downloaded: {}".format(image_name))
                
            except subprocess.CalledProcessError as e:
                error_msg = "Singularity build failed for {}: {}".format(
                    image_name, 
                    e.stderr.decode() if e.stderr else str(e)
                )
                raise ImageDownloadError(error_msg)
            except subprocess.TimeoutExpired:
                raise ImageDownloadError("Timeout building image: {}".format(image_name))
            except (OSError, shutil.Error) as e:
                raise ImageDownloadError("File operation failed for {}: {}".format(image_name, e))
    
    def update_symlink(self, symlink_path, target_path):
        """Update or create symlink."""
        # Ensure parent directory exists
        symlink_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Remove existing symlink if it exists
        if symlink_path.exists():
            symlink_path.unlink()
        
        # Create new symlink
        symlink_path.symlink_to(target_path)
        
        self.logger.info("Updated symlink: {} -> {}".format(symlink_path, target_path))
    
    def process_image(self, image_name):
        """Process a single image."""
        if not isinstance(image_name, str):
            raise ConfigError("Image name must be a string, got: {}".format(type(image_name).__name__))
        
        try:
            self.logger.info("Processing image: {}".format(image_name))
            
            # Get paths
            symlink_path = self.get_symlink_path(image_name)
            
            # Get current hash from symlink
            current_hash = self.get_current_hash_from_symlink(symlink_path)
            old_hash_dir = None
            if current_hash:
                old_hash_dir = self.get_hash_directory(current_hash)
            
            # Get remote hash
            self.logger.info("Fetching manifest for: {}".format(image_name))
            remote_hash = self.get_image_manifest_hash(image_name)
            
            # Compare hashes
            if current_hash == remote_hash:
                self.logger.info("Image {} is up to date (hash: {})".format(image_name, current_hash))
                return
            
            # Need to download
            if current_hash:
                self.logger.info("Image {} needs update: {} -> {}".format(
                    image_name, current_hash[:16] + '...', remote_hash[:16] + '...'
                ))
            else:
                self.logger.info("Image {} is new, downloading (hash: {})".format(
                    image_name, remote_hash[:16] + '...'
                ))
            
            # Get hash directory
            hash_dir = self.get_hash_directory(remote_hash)
           
            # Start transaction if the destination dir is CVMFS
            self.start_txn(hash_dir)
             
            # Download image
            self.download_image(image_name, hash_dir)
            
            # Update symlink
            # Compute relative path from symlink to target
            try:
                relative_target = os.path.relpath(str(hash_dir), str(symlink_path.parent))
                self.update_symlink(symlink_path, relative_target)
            except ValueError:
                # Fallback to absolute path if relative path computation fails
                self.update_symlink(symlink_path, hash_dir)
            
            # Publish repo if the destination dir is CVMFS
            self.publish_txn(hash_dir)
 
            # Remove old hash directory after successful symlink update
            if old_hash_dir and old_hash_dir.exists() and old_hash_dir != hash_dir:
                try:
                    shutil.rmtree(str(old_hash_dir))
                    self.logger.info("Removed old hash directory: {}".format(old_hash_dir))
                except (OSError, shutil.Error) as e:
                    self.logger.warning("Failed to remove old hash directory {}: {}".format(old_hash_dir, e))
            
            self.logger.info("Successfully processed image: {}".format(image_name))
            
        except (ManifestFetchError, ImageDownloadError) as e:
            self.logger.error("Failed to process image {}: {}".format(image_name, e))
            raise
        except Exception as e:
            self.logger.error("Unexpected error processing image {}: {}".format(image_name, e))
            raise ImageUnpackerError("Unexpected error: {}".format(e))
    
    def remove_unlisted_images(self, current_images, singularity_base, test=False):
        """
        Remove the images that are not in the current list
        """
        # Get all the image paths
        named_image_dirs = set()
        for subdir, dirs, files in os.walk(singularity_base):
            try:
                images_index = dirs.index(".images")
                del dirs[images_index]
            except ValueError as ve:
                pass
            for directory in dirs:
                path = os.path.join(subdir, directory)
                if os.path.islink(path):
                    named_image_dirs.add(path)

        # Compare the list of current images with the list of images from the FS
        for image in current_images:
            # Always has the registry as the first entry, remove it
            image_dir = image.split('/', 1)[-1]
            full_image_dir = os.path.join(singularity_base, image_dir)
            if full_image_dir in named_image_dirs:
                named_image_dirs.remove(full_image_dir)

        # named_image_dirs should now only contain containers that are
        # not in the images
        if len(named_image_dirs) > 0:
            self.start_txn(singularity_base)
            for image_dir in named_image_dirs:
                target_path = os.path.realpath(image_dir)
                #print("The target path of image is  %s" % target_path)
                print("Removing deleted image: %s" % image_dir)
                if not test:
                    try:
                        os.unlink(image_dir)
                        shutil.rmtree(target_path)
                    except OSError as e:
                        print("Failed to remove deleted image: %s" % e)
            self.publish_txn(singularity_base)

    def start_txn(self, hash_dir):
        global _in_txn
        if str(hash_dir).startswith("/cvmfs/"):
            cvmfs_repo = (str(hash_dir).split('/'))[2]
            print ("cvmfs_repo: " + cvmfs_repo)
            if _in_txn:
                return 0
            if os.path.exists("/var/spool/cvmfs/" + cvmfs_repo + "/in_transaction.lock"):
                result = os.system("cvmfs_server abort -f " + cvmfs_repo)
                print ("abort result: " + str(result))
                if result:
                    self.logger.error ("Failed to abort lingering transaction of repo " + cvmfs_repo + "(exit status: " + istr(result) + ")" )
                    return 1
            result = os.system("cvmfs_server transaction " + cvmfs_repo)
            if result:
                self.logger.error ("Transaction start failed (exit status : " + str(result) + "); will not attempt update.")
                return 1
        _in_txn = True

    def publish_txn (self, hash_dir):
        global _in_txn
        if str(hash_dir).startswith("/cvmfs/"):
            cvmfs_repo = (str(hash_dir).split('/'))[2]
            if _in_txn:
                _in_txn = False
                return os.system("cvmfs_server publish " + cvmfs_repo)
        else:
            _in_txn = False
        return 0
            
    def run(self):
        """Main execution method."""
        try:
            # Load images from remote
            self.images = self.load_images_from_remote(self.remote_url)
            
            # Check Singularity availability
            try:
                subprocess.run(['singularity', '--version'], 
                             stdout=PIPE, stderr=PIPE, check=True)
            except (subprocess.CalledProcessError, FileNotFoundError):
                raise ImageUnpackerError("Singularity/Apptainer not found or not working")
            
            # Ensure image directory exists
            self.image_dir.mkdir(parents=True, exist_ok=True)
            
            # Process each image
            total_images = len(self.images)
            self.logger.info("Starting to process {} images".format(total_images))
            
            errors = []
            for i, image_name in enumerate(self.images, 1):
                try:
                    self.logger.info("Processing image {} of {}".format(i, total_images))
                    self.process_image(image_name)
                except Exception as e:
                    errors.append((image_name, str(e)))
                    continue

            # Remove images that are not in the list
            self.remove_unlisted_images(self.images, self.image_dir)

            # Report results
            if errors:
                self.logger.error("Completed with {} errors:".format(len(errors)))
                for image_name, error in errors:
                    self.logger.error("  {}: {}".format(image_name, error))
                return 1
            else:
                self.logger.info("Successfully processed all {} images".format(total_images))
                return 0
                
        except Exception as e:
            self.logger.error("Fatal error: {}".format(e))
            return 1

    def verify(self):
        """To verify that images exist, no need to download"""
        try:
            # Load images from remote
            self.images = self.load_images_from_remote(self.remote_url)
            
            # Verify each image
            total_images = len(self.images)
            self.logger.info("Starting to verify {} images".format(total_images))

            errors = []
            for i, image_name in enumerate(self.images, 1):                
                try:
                    # Get remote hash
                    self.logger.info("Fetching manifest for: {}".format(image_name))
                    remote_hash = self.get_image_manifest_hash(image_name) 
                except Exception as e:
                    errors.append((image_name, str(e)))
                    continue
            # Report results
            if errors:
                self.logger.error("verified with {} errors:".format(len(errors)))
                for image_name, error in errors:
                    self.logger.error("  {}: {}".format(image_name, error))
                return 1
            else:
                self.logger.info("Successfully verified all {} images".format(total_images))
                return 0
                
        except Exception as e:
            self.logger.error("Fatal error: {}".format(e))
            return 1

_in_txn = False
def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Download and unpack OCI container images using Singularity',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Example remote file format (one image per line):
  docker.io/nginx:latest
  docker.io/busybox:latest
  quay.io/biocontainers/blast:2.9.0
  # This is a comment - lines starting with # are ignored
  tensorflow/tensorflow:latest

Example usage:
  python3 %(prog)s /tmp/cvmfs/images https://raw.githubusercontent.com/user/repo/main/images.txt
        '''
    )
    
    parser.add_argument(
        'image_dir',
        help='Directory where images will be stored'
    )
    
    parser.add_argument(
        'remote_url',
        help='URL to remote file containing list of images (one per line)'
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version='OCI Image Unpacker 1.0'
    )
    
    parser.add_argument(
        '--verify',
        #type=str,
        action='store_true',
        help='To verify the images exsistance without downloading'
    )
    
    args = parser.parse_args()
    
    try:
        unpacker = ImageUnpacker(args.image_dir, args.remote_url)
        if args.verify:
            return unpacker.verify()
        else:
            return unpacker.run()
    except KeyboardInterrupt:
        print("\nInterrupted by user", file=sys.stderr)
        return 130
    except Exception as e:
        print("Fatal error: {}".format(e), file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())

