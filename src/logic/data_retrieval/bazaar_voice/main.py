from argparse import ArgumentParser
from reviews.logger import setup_logging
from reviews.orchestrator import main


if __name__ == "__main__":
    parser = ArgumentParser(description="Fetch Bazaarvoice product reviews.")
    parser.add_argument('--brand', help="Brand name to query.", required=True)
    parser.add_argument('--max-workers', type=int, default=8)
    parser.add_argument('--limit', type=int, default=100)
    parser.add_argument('--verbose', action='store_true')

    args = parser.parse_args()

    setup_logging(args.verbose)
    main(brand=args.brand, max_workers=args.max_workers, limit=args.limit)