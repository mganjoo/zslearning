# Rakefile to download a few datasets and configure their locations.
#
# INSTRUCTIONS:
# 1. Please run this after obtaining Kerberos tickets so that SSH access
#    can proceed without any password requirement. Alternatively, enter your
#    SSH password when prompted.
# 2. If you run this on one of the FarmShare clusters, then it will create a symbolic
#    link to the data directories instead of downloading them. This is useful if you
#    want to save space on your AFS.

# Configuration information
CIFAR_FILE      = "cifar-10-matlab.tar.gz"
CIFAR_URL       = "http://www.cs.toronto.edu/~kriz/#{CIFAR_FILE}"
DATA_LOCAL_URL  = "/mnt/glusterfs/mganjoo/"
DATA_SSH_URL    = "corn.stanford.edu:#{DATA_LOCAL_URL}"
TRAIN_FILENAME = "train.mat"
TEST_FILENAME = "test.mat"

# File and directory names
IMAGE_DATA_DIR  = "image_data"
WORD_DATA_DIR   = "word_data"

desc "Directory for image data"
directory IMAGE_DATA_DIR

desc "Check if symlinks can be created"
task :check_symlinks do
    # Symlink if the data directory exists locally on the machine
    if File.directory? DATA_LOCAL_URL
        if !File.symlink? "#{DATA_LOCAL_URL}#{IMAGE_DATA_DIR}"
            puts "Symlinking image directory"
            File.symlink("#{DATA_LOCAL_URL}#{IMAGE_DATA_DIR}", "#{IMAGE_DATA_DIR}")
        end
        if !File.symlink? "#{DATA_LOCAL_URL}#{WORD_DATA_DIR}"
            puts "Symlinking word directory"
            File.symlink("#{DATA_LOCAL_URL}#{WORD_DATA_DIR}", "#{WORD_DATA_DIR}")
        end
    else
        Rake::Task["download_image_set"].invoke
        Rake::Task["download_word_set"].invoke
    end
end

desc "Download the image samples"
task :download_image_set => IMAGE_DATA_DIR do
    Dir.chdir(IMAGE_DATA_DIR) do
        if !File.exist? CIFAR_FILE
            puts "Getting CIFAR dataset"
            `wget #{CIFAR_URL}`
        end
        `tar -xzf #{CIFAR_FILE}`
        if !File.exist? TRAIN_FILENAME
            puts "Getting train data"
            `scp #{DATA_SSH_URL}#{IMAGE_DATA_DIR}/#{TRAIN_FILENAME} .`
        end
        if !File.exist? TEST_FILENAME
            puts "Getting test data"
            `scp #{DATA_SSH_URL}#{IMAGE_DATA_DIR}/#{TEST_FILENAME} .`
        end
    end
end

desc "Download the word vectors"
task :download_word_set do
    if !File.directory? WORD_DATA_DIR
        puts "Getting word data"
        `scp -r #{DATA_SSH_URL}#{WORD_DATA_DIR} ./#{WORD_DATA_DIR}`
    end
end

task :default => [ :check_symlinks ]