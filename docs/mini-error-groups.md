# Where the mini still fails

This is an error analysis of the candidate mini (`ema 0.999 + smooth_weight 1 + page_f1 4`,
`docs/objective.md`) on WMB val732 and on the Common Crawl page set (`results/ccbench.md`).

## WMB val: the worst 10 % of pages

Probabilities are averaged over the candidate's 3 seeds at the 0.5 threshold, which gives val F1
0.8791. Pages are ranked by room, the oracle DOM F1 minus the model's F1. The 73 worst of 732
pages hold 59 % of all room.

Per seed, those 73 pages hold 58 to 60 % of the room, and the shipped mini fails on the same pages.
So the losses come from the kind of page, not from training noise.

    python vendors/wmb/adapter/errors.py --worst s0/val.jsonl s1/val.jsonl s2/val.jsonl

The tool prints the ranked list. The grouping below was done by hand from the pages.

Columns:
- **F1 / oracle:** the model's F1, then the best F1 WMB's own labels allow on that page.
- **dir:** the side that loses more characters. FP means the model selected too much, FN that it
  missed gold.
- **sel/gold:** selected characters over gold characters.

| group | pages | share of the worst-page room | main error | remedy |
|---|---|---|---|---|
| A. Catalogue and listing pages | 16 | 23 % | FN: product grid dropped; FP: teaser lists kept | more training pages |
| B. Reference and bibliographic records | 10 | 15 % | FN: long citation lists dropped | more training pages |
| C. Forums, threads, Q&A, reviews | 13 | 16 % | mixed: quoted posts, footers, later replies | more pages, once the label policy is settled |
| D. Detail pages (product, property, event, calculator) | 15 | 19 % | FP: terms, price tables, other listings | more training pages |
| E. Article plus tail (teasers, bios, disclaimers) | 8 | 10 % | FP: whole following articles selected | more training pages |
| F. Junk render (JSON config, PHP dumps, wiki diffs) | 4 | 7 % | FP: blobs selected | render fix in the package |
| G. Short and sparse pages | 7 | 9 % | FN: tiny gold, mostly policy | none: label noise |

What the val pages show about WMB's labels:
- A product or category listing's grid is main content: names, prices and short blurbs.
- A forum thread's posts are main content, but WMB is inconsistent about replies (group C).
- Site footers, terms of service, "related" teaser lists and sort or filter forms are not main
  content.

### A. Catalogue and listing pages (16)

These are category pages of small shops, directory and listing pages, and "top stories" indexes.

- **FN:** the model keeps the intro paragraph and drops the product table or grid. The most
  common case is a product table with name, price and order columns.
- **FP:** the model keeps the sidebar or the teaser lists of other categories.

Typical sources:
- small-shop category pages: osCommerce, ColdFusion `.cfm`, Magento, WooCommerce
  `product-category/`;
- business directories: `/directories/`, `/listing/`;
- news index pages.

| # | url | F1 / oracle | dir | sel/gold |
|---|---|---|---|---|
| 2 | https://shop.americandigitals.com/cell-phones-accessories-product-line.html | 0.02 / 1.00 | FN | 0.0 |
| 12 | http://www.simplepastimes.com/cat_unique.cfm | 0.01 / 0.84 | FN | 0.1 |
| 14 | http://www.metrokitchen.com/category/wellness-kitchen-anti-fatigue-mats | 0.19 / 1.00 | FN | 0.1 |
| 13 | http://www.gamemodding.net/en/world-of-tanks/wot-skins/?wot_replace_model=Somua%2bSAu-40 | 0.18 / 1.00 | FN | 0.2 |
| 24 | http://www.simplepastimes.com/cat_landscapes.cfm | 0.14 / 0.83 | FN | 0.2 |
| 22 | http://www.simplepastimes.com/cat-wentworth-wooden-puzzles4.cfm | 0.16 / 0.88 | FN | 0.3 |
| 29 | http://www.simplepastimes.com/cat-cobble-hill4.cfm | 0.25 / 0.86 | FN | 0.3 |
| 26 | https://www.medalsandmemorabilia.com/product-category/orders-medals-decorations/casualty-medals-and-plaques/ | 0.31 / 1.00 | FN | 0.2 |
| 23 | http://www.bergen.com/top-stories | 0.28 / 1.00 | FP | 5.5 |
| 28 | https://www.latiendadelyoyo.com/en/17_i-love-yoyo | 0.35 / 1.00 | FP | 4.1 |
| 63 | http://www.nutritionexpress.com/protein/plant+protein?p=2 | 0.33 / 0.70 | FP | 3.9 |
| 33 | http://www.madisonmagazine.com/Madison-Magazine/Food-Dining/Dining-Guide/index.php/name/Orient-House/listing/34391/ | 0.42 / 1.00 | FP | 2.6 |
| 43 | http://www.freenew.net/windows8/dr-who-companion/940.htm | 0.53 / 1.00 | FP | 2.1 |
| 53 | https://www.churchfinder.com/churches/wa/spokane | 0.59 / 1.00 | FP | 2.1 |
| 56 | http://www.bioanalyticsystems.com/html/list-15-2.html | 0.61 / 1.00 | FP | 1.6 |
| 65 | https://ambalco.com/cat/city+bike/bike+commute+challenge/?p=6 | 0.64 / 0.99 | FN | 0.6 |

### B. Reference and bibliographic records (10)

These are author pages that list papers (RePEc, university research portals), Google Books
records, dictionary entries, surname and genealogy pages, glossaries and authority records.

- **FN:** the page's body is a long list of short, citation-like rows. They look like
  navigation, so the model drops them.
- **FP:** the model keeps an "activities" or "contents" navigation list instead. 1828.mshaffer.com
  is a PHP error dump and belongs with F.

| # | url | F1 / oracle | dir | sel/gold |
|---|---|---|---|---|
| 0 | https://ideas.repec.org/e/pfa184.html | 0.00 / 1.00 | FN | 0.0 |
| 3 | https://books.google.com.vn/books?id=2Sc2AAAAMAAJ&hl=vi&source=gbs_book_other_versions_r&cad=3 | 0.03 / 1.00 | FN | 0.0 |
| 8 | https://scholars.ln.edu.hk/en/persons/ka-ho-joshua-mok/activities/ | 0.12 / 1.00 | FP | 17.2 |
| 21 | http://www.histopolis.com/Surname/Huski | 0.27 / 1.00 | FN | 0.2 |
| 19 | https://1828.mshaffer.com/d/word/guide/simple/ | 0.25 / 1.00 | FP | 7.4 |
| 38 | http://ideas.repec.org/h/erf/erfssc/15-1.html | 0.48 / 0.99 | FN | 0.3 |
| 35 | https://www.originalsources.com/Document.aspx?DocID=54U8SJVTZB3GDTW | 0.43 / 1.00 | FP | 3.4 |
| 44 | https://digitalcommons.usf.edu/gandy/index.11.html | 0.34 / 0.81 | FN | 0.4 |
| 45 | https://www.eac.gov/glossary/?Letter=H | 0.39 / 0.85 | FP | 2.6 |
| 68 | https://kulturnav.org/f0b103b1-f031-42e0-bbed-a114cefb4b8d | 0.65 / 0.98 | FP | 1.4 |

### C. Forums, threads, Q&A, reviews (13)

These are table-layout forums (SMF, vBulletin, `viewit.cgi`), Chinese and Hong Kong BBS, Q&A
pages, review pages and blog comment sections. Modern Discourse-style threads are not among them.

- **FN:** the model misses the replies after the first post, or posts inside a table layout.
- **FP:** the model keeps quoted copies of the post (bbs1.people.com.cn repeats the body), board
  footers and leaderboards.
- **Label policy:** WMB is inconsistent here. Some threads label only the first post, others
  every reply.

| # | url | F1 / oracle | dir | sel/gold |
|---|---|---|---|---|
| 10 | http://www.bigchurch.com/blog/index.html?who=r,JUzSjjsg8ulBQADN37PyCX6nok_WGc0N4xajGJ0mkoWNIFMmJHVq9iXTt_c4PKyNyNhJwtxbkKrLJOuNRQuVJsXyEOsWWYtsIFtnqc7Smyz_Bub_js598aWKzzkeQTYZ | 0.03 / 0.88 | FN | 0.0 |
| 7 | https://rc.runryder.com/gallery/62757/ | 0.11 / 1.00 | FN | 0.1 |
| 15 | http://bbs1.people.com.cn/post/2/0/0/169316006.html | 0.19 / 1.00 | FP | 10.2 |
| 20 | http://www.yesterdaystractors.com/cgi-bin/viewit.cgi?bd=ttalk&th=1252948 | 0.11 / 0.84 | FP | 16.6 |
| 32 | https://www.paopao.info/#/post?id=1080035365 | 0.40 / 0.99 | FP | 3.2 |
| 39 | https://answers.angieslist.com/Who-leak-detection-Bay-Window-repair-leak-slight-water-damage-interior-dry-wall-q197578.aspx | 0.46 / 0.97 | FN | 0.9 |
| 52 | https://www.strahlen.org/forum/index.php?topic=3058.msg90778 | 0.57 / 1.00 | FN | 0.5 |
| 54 | http://www.hotelcurious.com/eng/blog-hotel-curious-barcelona/freestyle_mx_masters_palau_sant_jordi/158 | 0.59 / 1.00 | FP | 2.2 |
| 55 | https://lihkg.com/thread/1551476/page/27 | 0.59 / 1.00 | FP | 1.5 |
| 57 | http://dibujando.net/foro/hola-desde-mexico-32492 | 0.61 / 1.00 | FN | 1.0 |
| 59 | https://www.klantenvertellen.nl/reviews/1039657/marlog_car_handling?filterLocale=en&lang=en | 0.62 / 1.00 | FP | 1.8 |
| 61 | https://historyhub.history.gov/military-records/army-and-air-force-records/f/army-air-force-records-forum/16424/seeking-roster-for-45th-inf-division-ww2/87967 | 0.63 / 1.00 | FN | 0.5 |
| 72 | https://bbs.nanshengbbs.top/topic/detail/1102 | 0.69 / 1.00 | FP | 1.9 |

### D. Detail pages: product, property, event, calculator (15)

Gold is short on these pages (title, price, specs), so anything extra dominates the score.

- **FP:** the model keeps delivery terms (jacamo), a season price table (erento), "other
  properties" (apartamenty), other suppliers in a directory (no-tillfarmer) and disclaimers
  (planetdrugs).
- **FN:** the model misses calculator result tables (dollartimes) and spec blocks.

| # | url | F1 / oracle | dir | sel/gold |
|---|---|---|---|---|
| 4 | https://www.no-tillfarmer.com/directories/95-guide-to-no-till-suppliers/listing/305-dairyland-seed-co-inc | 0.04 / 1.00 | FP | 4.0 |
| 5 | https://www.dollartimes.com/loans/mortgage-payment.php?length=30&amount=397500&rate=23.25 | 0.06 / 1.00 | FN | 0.1 |
| 18 | https://www.haya.es/promocion-obra-nueva/promocion-1000011534/ | 0.23 / 1.00 | FN | 0.2 |
| 25 | https://sazalem.com/songs/zholdasbekabdihanov-menseniumytpadym | 0.31 / 1.00 | FP | 5.0 |
| 34 | http://www.jacamo.co.uk/shop/union-blues-button-fly-bootcut-jeans-33in/kk654/product/details/show.action?pdBoUid=1883 | 0.42 / 1.00 | FP | 4.1 |
| 36 | https://www.apartamenty.kz/en/pavlodar/apartments/1bedroom-apartment-18917.html | 0.46 / 0.99 | FP | 3.0 |
| 42 | https://wemakeit.com/users/nancy-3/show/backed-projects | 0.53 / 1.00 | FP | 1.8 |
| 47 | https://argohosting.net/kvm-vps-hosting/ | 0.55 / 1.00 | FN | 0.3 |
| 48 | https://www.planetdrugsdirect.com/medications/hot-flushes | 0.55 / 1.00 | FP | 2.7 |
| 51 | http://www.coolhouseplans.com/details.html?pid=chp-17594 | 0.54 / 0.98 | FP | 2.3 |
| 62 | http://www.erento.com/mieten/ferienwohnung_ferienhaus/griechenland/kreta/4662159287-price.html?location=154066 | 0.63 / 1.00 | FP | 2.1 |
| 64 | http://thewalters.org/events/eventdetails.aspx?e=2960 | 0.64 / 1.00 | FP | 2.1 |
| 66 | https://www.truthfinder.com/report-review/?firstName=Melissa&lastName=Pantoja&state=AZ&pid=4:eJyKVvJNzcksLk5U0lFS0lEKSMwryc8CcaDCCggRNGRhbuKlGxFg4q6ko1RSlJhXXJqXmZ-npKOUZJxiYZJskaiblGpprGuSmJKqa5mWlKRrlGKcamGalmRonmyiFAsIAAD__zOwIhw= | 0.63 / 0.98 | FN | 0.5 |
| 67 | https://www.foreca.com/100179525/Taveta-Taita%20Taveta,%20Kenya/overview | 0.66 / 1.00 | FN | 0.5 |
| 71 | http://www.layercakeshop.com/collections/party/products/gift-card | 0.68 / 1.00 | FP | 1.7 |

### E. Article plus tail (8)

An article is followed by full-text teasers of other articles, an author bio or a disclaimer,
all inside the same content container.

The tail looks exactly like the article, so it is all selected: 2 to 29 times the gold. Typical
sources:
- news and magazine sites that render the next article inline (innpoland, dicardiology);
- government topic pages with event blurbs;
- aphorism sites;
- community profile pages;
- investment newsletters with long legal disclaimers.

intomore is a WMB oddity: its gold includes the cookie-policy table.

| # | url | F1 / oracle | dir | sel/gold |
|---|---|---|---|---|
| 11 | http://www.dicardiology.com/videos/video-editors-choice-most-innovative-new-cardiac-ultrasound-ase-2016 | 0.05 / 0.89 | FP | 29.0 |
| 9 | https://innpoland.pl/197155,dziedziczenie-srodkow-na-subkoncie-w-zus-po-zmarlym-mozna-zyskac-fortune | 0.14 / 1.00 | FP | 12.7 |
| 31 | https://www.climatehubs.oce.usda.gov/climate-impacts/drought | 0.40 / 1.00 | FP | 4.3 |
| 41 | https://www.aphorism.it/aforismi/l-amore-non-dichiarato-e-come-una-spina-conficc/ | 0.52 / 1.00 | FP | 2.8 |
| 46 | https://www.futuremoneytrends.com/strategic-esg-investment-position-were-adding-this-one-early-and-often | 0.54 / 1.00 | FP | 2.8 |
| 49 | http://vegsource.com/joel-fuhrman-md/ | 0.56 / 1.00 | FP | 2.5 |
| 50 | https://www.intomore.com/entertainment/tv/was-this-the-gayest-greys-anatomy-season-ever/ | 0.56 / 1.00 | FN | 0.4 |
| 58 | http://medianetwerk.ning.com/profile/billyjonesbluez?xg_source=profiles_memberList | 0.60 / 0.98 | FP | 2.2 |

### F. Junk render (4)

These pages are fixed in the render, not by training.

- **zhihu (2 pages):** a JSON client config inside `<textarea>` is rendered and selected, 200
  times the gold on one page. Skipping `textarea`, `template` and `noscript` in the render fixes
  it, and measured +0.003 val on every seed.
- **buste.in:** a wiki backup diff in a code block.
- **trac.i2p2.de:** a tag dropdown rendered as a list.

| # | url | F1 / oracle | dir | sel/gold |
|---|---|---|---|---|
| 1 | https://zhuanlan.zhihu.com/shenzhenware/about | 0.01 / 1.00 | FP | 202.7 |
| 6 | https://buste.in/kousoku/index.php?cmd=backup&page=Usher%20Tickets%20Omg%20Tour&age=3&action=nowdiff | 0.07 / 1.00 | FP | 16.8 |
| 17 | https://zhuanlan.zhihu.com/p/26614314 | 0.22 / 1.00 | FP | 9.5 |
| 60 | https://trac.i2p2.de/browser/debian/po?rev=a45688867d3017144a2edfe52766e8159e710c68&order=date | 0.62 / 1.00 | FP | 2.0 |

### G. Short and sparse pages (7)

These are galleries, polls, course pages and pages with tiny gold. Gold is one or two lines,
often with an oracle below 1, and which line is gold is close to arbitrary.

| # | url | F1 / oracle | dir | sel/gold |
|---|---|---|---|---|
| 16 | http://ranganna.com/cursai.aspx?id=3&lang=en | 0.19 / 1.00 | FN | 0.3 |
| 27 | http://archive.clarionledger.com/poll/2014-02-11/7791859/results?omnivar=event53&from=global | 0.22 / 0.89 | FP | 1.0 |
| 30 | http://lauramillerartist.com/farmers-market-veggies-sketches/6farmers-market-veggies/ | 0.39 / 1.00 | FN | 0.2 |
| 37 | http://gaokao.koolearn.com/college/s/081301 | 0.44 / 0.96 | FN | 0.3 |
| 40 | http://www.abrazohealth.com/services/mothernewborncare/treatments/resources.aspx?chunkiid=127483 | 0.51 / 1.00 | FN | 0.5 |
| 69 | http://www.taylorpictures.net/thumbnails.php?album=815 | 0.49 / 0.81 | FN | 0.6 |
| 70 | https://www.tattoonow.com/gallery/tattoos/artist/hyeonhwa-kim-dokhwa~8978/lost-in-translation-tattoo~143373 | 0.51 / 0.83 | FP | 1.4 |

## Common Crawl: empty texts

On the 7,177 Common Crawl pages, with the title dropped and repeated lines removed, the 0.3.0
mini returns an empty text on 637 pages. trafilatura returns an empty text on 59.

Most of the gap is not a model failure:
- 322 pages are bot-challenge or JavaScript-wall interstitials. They were captured by the crawler
  in place of the page.
- trafilatura extracts the challenge line, typically "Please wait while your request is being
  verified..." (13 tokens). That line passes the DCLM filter on 305 of these pages.
- Returning nothing on them is the right output. The 0.3.0 mini does so on 319 of the 322.
- The detection rule: trafilatura text of 60 tokens or fewer that contains a challenge phrase,
  case-insensitive.
  The phrases are "request is being verified", "testing whether you are a human", "The requested
  URL was rejected" and "enable JavaScript".

The remaining 318 empty pages (2,388 weighted) are the model's real misses. trafilatura's text on
them is empty on 3, 20 tokens or fewer on 26, over 200 tokens on 72, median 73 tokens; it passes
DCLM on 59 of them and Edu on 1.

Whether these pages are empty by content, with trafilatura picking up a stray line, or real
content the model drops, is not yet known. That needs a hand label per page.
