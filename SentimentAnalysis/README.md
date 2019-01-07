<h1> Proof-of-Consent for building batch Recommendation Systems. </h1>

<p>
    <h3> Algorithm Used </h3>
    <p> SentiWordNet Interface - NLTK</p>
    <p> Reference: http://www.nltk.org/howto/sentiwordnet.html </p>

    <h3> Frameworks Used </h3>
    <p> >> Numpy </p>
    <p> >> Pandas </p>
    <p> >> Matplotlib </p>

    <h3> Dataset used </h3>
    <p> MARD: Multimodal Album Reviews Dataset <p>
    <p> Reference: https://www.upf.edu/web/mtg/mard </p>

    <h4> Sources </h4>
    <p> Amazon </p>
    <p>
        amazon-id: The Amazon product id. You can visualize the album page in amazon adding this id to the following url "www.amazon.com/dp/"
        artist: The artist name as it appears in Amazon
        title: The album title as it appears in Amazon related:
           also bought: Other products bought by people who bought this album
           buy_after_viewing: Other products bought by people after viewing this album
        price: The album price
        label: The record label of the album
        categories: The genre categories in Amazon
        sales_rank: Ranking in the Amazon music sales rank
        imUrl: Url of the album cover
        artist_url: The url of the artist page in amazon. You must add "www.amazon.com" at the beginning to access this page
        root-genre: The root genre category of the album, extracted from the categories field.
    </p>

    <p> MusicBrainz </p>
    <p>
        artist-mbid: The MusicBrainz ID of the artist
        first-release-year: The year of first publication of the album
        release-group-mbid: The MusicBrainz ID of the release group mapped to this album title
        release-group: The MusicBrainz ID of the first release in the release-group of this album, used to extract the tracks info
        songs: List of tracks in the album
        ftitle: Title of the track
        fmbid: MusicBrainz recording ID of the track. Used to map with AcousticBrainz.
    </p>
</p>