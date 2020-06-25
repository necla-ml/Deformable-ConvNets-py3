all:
	# cd nms/; python setup_linux.py build_ext --inplace; rm -rf build; cd ../../
	# cd bbox/; python setup_linux.py build_ext --inplace; rm -rf build; cd ../../
	# cd dataset/pycocotools/; python setup_linux.py build_ext --inplace; rm -rf build; cd ../../
	# python setup.py build_ext --inplace; rm -rf build
	pip install -e .

clean:
	python setup.py clean --all
	cd dcn/bbox; rm -fr *.so *.c *.cpp
	cd dcn/nms; rm -fr *.so *.c *.cpp
	cd dcn/dataset/pycocotools/; rm -fr _mask.c *.sou

## VCS

require-version:
ifndef version
	$(error version is undefined)
endif

tag: require-version
	git checkout master
	git tag -a v$(version) -m v$(version) 
	git push origin tags/v$(version)

del-tag:
	git tag -d $(tag)
	git push origin --delete tags/$(tag)