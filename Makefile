all:
	# cd nms/; python setup_linux.py build_ext --inplace; rm -rf build; cd ../../
	# cd bbox/; python setup_linux.py build_ext --inplace; rm -rf build; cd ../../
	# cd dataset/pycocotools/; python setup_linux.py build_ext --inplace; rm -rf build; cd ../../
	python setup.py build_ext --inplace; rm -rf build

clean:
	python setup.py clean --all
	cd dcn/bbox; rm -fr *.so *.c *.cpp
	cd dcn/nms; rm -fr *.so *.c *.cpp
	cd dcn/dataset/pycocotools/; rm -fr _mask.c *.sou