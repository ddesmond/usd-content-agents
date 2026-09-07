Deliberately empty.

`docker-compose.yml` mounts `${OVRTX_VMATERIALS_DIR:-./empty-vmaterials}` onto
the renderer's MDL search root at `bin/library/mdl/mdl/vMaterials_2`. Set
`OVRTX_VMATERIALS_DIR` to an installed vMaterials 2 tree to make
`@vMaterials_2/<Category>/<Module>.mdl@` resolve; leave it unset and this
directory mounts instead, which changes nothing.

Compose has no conditional volumes, so the default has to be a real path. This
is that path. Do not put anything else in it.
