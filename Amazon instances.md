# Amazon instances

## Connect via ssh

default user on Amazon linux `ec2-user`

`ssh -i my_1.pem user@ec2-ip.dns`

## Pip 

#### `locale.Error: unsupported locale setting`

`sudo apt-get install language-pack-en-base`

`export LC_CTYPE="en_US.UTF-8"`

`export LC_ALL=C`

`sudo pip install ipython`

#### `sudo: pip: command not found`

the path to pip (usually `/usr/local/bin`) is not in the sudo path by defalt so add it : `cd /etc/` puis edit (vi ou nano) `sudo nano sudoers` et ajouter `/usr/local/bin` au `secure_path` de la  la ligne `Defaults    secure_path = /sbin:/bin:/usr/sbin:/usr/bin` -> `Defaults    secure_path = /sbin:/bin:/usr/sbin:/usr/bin:/usr/local/bin`

## Copy 

### File

`scp -i my_1.pem my_file.ext user@ec2-ip.dns:location_on_instance/`

### Directory

`scp -rp -i my_1.pem my_dir/ user@ec2-ip.dns:location_on_instance/`
