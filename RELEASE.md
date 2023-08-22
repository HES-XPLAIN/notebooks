# Release

Instructions for the release process and publication on ghcr.io.

## Create new release

In *Releases*, select *Draft a new release* in the GitHub interface.

* In *Choose a tag*, enter a new tag in the form `X.Y.Z`
* Add notes to describe the changes
* Select *Publish release*

Alternatively, use the GitHub CLI interactively:
```
gh release create
```

## Set visibility to public

In *Package* > *Package settings* > *Danger Zone*, ensure the package visibility
is set to **Public**.

## Done!

Wait a few minutes for the image to be automatically built and published on ghcr.io.
