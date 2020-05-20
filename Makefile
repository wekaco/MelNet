build:
	docker build --force-rm -t wekaco/${shell basename "${PWD}" | awk '{print tolower($0)}' }:${shell git rev-parse --abbrev-ref HEAD | sed 's/\//-/g'} -f docker/Dockerfile .

test:
	docker run --volume ${shell pwd }:/app --rm -ti wekaco/${shell basename "${PWD}" | awk '{print tolower($0)}' }:${shell git rev-parse --abbrev-ref HEAD | sed 's/\//-/g'} python trainer.py -c ./novo.yaml -n test -t 1 -b 1
