#! /usr/bin/perl -w
#
# Copyright (C) 2012, Patrick Michl <p.michl (at) dkfz.de>

use strict;
use warnings;
use utf8;
use FindBin;                  # locate this script
use lib "$FindBin::Bin/lib";  # use the current lib directory
use metacore;                 # metacore external API
use Data::Dumper;

#create metacore class instance
my $mc = new metacore();
my $key = $mc->login('heidelberg5', '562412');
print "Key: $key\n";
my $version = $mc->getVersion();
print "MetaCore Version: $version\n";

print Dumper($mc->doRegulationSearch('a'));

$mc->logout();

