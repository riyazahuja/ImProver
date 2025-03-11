/-- The functor associating the *singular simplicial set* to a topological space.

Let `X` be a topological space.
Then the singular simplicial set of `X`
has as `n`-simplices the continuous maps `[n].toTop → X`.
Here, `[n].toTop` is the standard topological `n`-simplex,
defined as `{ f : Fin (n+1) → ℝ≥0 // ∑ i, f i = 1 }` with its subspace topology. -/
noncomputable def TopCat.toSSet : TopCat ⥤ SSet :=
  Presheaf.restrictedYoneda SimplexCategory.toTop


/-- The *geometric realization functor* is
the left Kan extension of `SimplexCategory.toTop` along the Yoneda embedding.

It is left adjoint to `TopCat.toSSet`, as witnessed by `sSetTopAdj`. -/
noncomputable def SSet.toTop : SSet ⥤ TopCat :=
  yoneda.leftKanExtension SimplexCategory.toTop


/-- Geometric realization is left adjoint to the singular simplicial set construction. -/
noncomputable def sSetTopAdj : SSet.toTop ⊣ TopCat.toSSet :=
  Presheaf.yonedaAdjunction (yoneda.leftKanExtension SimplexCategory.toTop)
    (yoneda.leftKanExtensionUnit SimplexCategory.toTop)


/-- The geometric realization of the representable simplicial sets agree
  with the usual topological simplices. -/
noncomputable def SSet.toTopSimplex :
    (yoneda : SimplexCategory ⥤ _) ⋙ SSet.toTop ≅ SimplexCategory.toTop :=
  Presheaf.isExtensionAlongYoneda _

