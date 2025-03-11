instance projective_ultrafilter (X : Type u) : Projective (of <| Ultrafilter X) where
  factors {Y Z} f g hg := by
    /-
      X : Type u
      Y Z : Profinite
      f : Quiver.Hom (Profinite.of (Ultrafilter X)) Z
      g : Quiver.Hom Y Z
      hg : CategoryTheory.Epi g
      ⊢ Exists fun f' => Eq (CategoryTheory.CategoryStruct.comp f' g) f
    -/
    rw [epi_iff_surjective] at hg
    /-
      X : Type u
      Y Z : Profinite
      f : Quiver.Hom (Profinite.of (Ultrafilter X)) Z
      g : Quiver.Hom Y Z
      hg : Function.Surjective ⇑g
      ⊢ Exists fun f' => Eq (CategoryTheory.CategoryStruct.comp f' g) f
    -/
    obtain ⟨g', hg'⟩ := hg.hasRightInverse
    /-
      case intro
      X : Type u
      Y Z : Profinite
      f : Quiver.Hom (Profinite.of (Ultrafilter X)) Z
      g : Quiver.Hom Y Z
      hg : Function.Surjective ⇑g
      g' : (CategoryTheory.forget Profinite).obj Z → (CategoryTheory.forget Profinit …
      hg' : Function.RightInverse g' ⇑g
      ⊢ Exists fun f' => Eq (CategoryTheory.CategoryStruct.comp f' g) f
    -/
    let t : X → Y := g' ∘ f ∘ (pure : X → Ultrafilter X)
    /-
      case intro
      X : Type u
      Y Z : Profinite
      f : Quiver.Hom (Profinite.of (Ultrafilter X)) Z
      g : Quiver.Hom Y Z
      hg : Function.Surjective ⇑g
      g' : (CategoryTheory.forget Profinite).obj Z → (CategoryTheory.forget Profinit …
      hg' : Function.RightInverse g' ⇑g
      t : X → ↑Y.toTop := Function.comp g' (Function.comp (⇑f) Pure.pure)
      ⊢ Exists fun f' => Eq (CategoryTheory.CategoryStruct.comp f' g) f
    -/
    let h : Ultrafilter X → Y := Ultrafilter.extend t
    /-
      case intro
      X : Type u
      Y Z : Profinite
      f : Quiver.Hom (Profinite.of (Ultrafilter X)) Z
      g : Quiver.Hom Y Z
      hg : Function.Surjective ⇑g
      g' : (CategoryTheory.forget Profinite).obj Z → (CategoryTheory.forget Profinit …
      hg' : Function.RightInverse g' ⇑g
      t : X → ↑Y.toTop := Function.comp g' (Function.comp (⇑f) Pure.pure)
      h : Ultrafilter X → ↑Y.toTop := Ultrafilter.extend t
      ⊢ Exists fun f' => Eq (CategoryTheory.CategoryStruct.comp f' g) f
    -/
    have hh : Continuous h := continuous_ultrafilter_extend _
    /-
      case intro
      X : Type u
      Y Z : Profinite
      f : Quiver.Hom (Profinite.of (Ultrafilter X)) Z
      g : Quiver.Hom Y Z
      hg : Function.Surjective ⇑g
      g' : (CategoryTheory.forget Profinite).obj Z → (CategoryTheory.forget Profinit …
      hg' : Function.RightInverse g' ⇑g
      t : X → ↑Y.toTop := Function.comp g' (Function.comp (⇑f) Pure.pure)
      h : Ultrafilter X → ↑Y.toTop := Ultrafilter.extend t
      hh : Continuous h
      ⊢ Exists fun f' => Eq (CategoryTheory.CategoryStruct.comp f' g) f
    -/
    use ⟨h, hh⟩
    /-
      case h
      X : Type u
      Y Z : Profinite
      f : Quiver.Hom (Profinite.of (Ultrafilter X)) Z
      g : Quiver.Hom Y Z
      hg : Function.Surjective ⇑g
      g' : (CategoryTheory.forget Profinite).obj Z → (CategoryTheory.forget Profinit …
      hg' : Function.RightInverse g' ⇑g
      t : X → ↑Y.toTop := Function.comp g' (Function.comp (⇑f) Pure.pure)
      h : Ultrafilter X → ↑Y.toTop := Ultrafilter.extend t
      hh : Continuous h
      ⊢ Eq (CategoryTheory.CategoryStruct.comp { toFun := h, continuous_toFun := hh  …
    -/
    apply (forget Profinite).map_injective
    /-
      case h.a
      X : Type u
      Y Z : Profinite
      f : Quiver.Hom (Profinite.of (Ultrafilter X)) Z
      g : Quiver.Hom Y Z
      hg : Function.Surjective ⇑g
      g' : (CategoryTheory.forget Profinite).obj Z → (CategoryTheory.forget Profinit …
      hg' : Function.RightInverse g' ⇑g
      t : X → ↑Y.toTop := Function.comp g' (Function.comp (⇑f) Pure.pure)
      h : Ultrafilter X → ↑Y.toTop := Ultrafilter.extend t
      hh : Continuous h
      ⊢ Eq ((CategoryTheory.forget Profinite).map (CategoryTheory.CategoryStruct.com …
    -/
    simp only [h, ContinuousMap.coe_mk, coe_comp]
    /-
      case h.a
      X : Type u
      Y Z : Profinite
      f : Quiver.Hom (Profinite.of (Ultrafilter X)) Z
      g : Quiver.Hom Y Z
      hg : Function.Surjective ⇑g
      g' : (CategoryTheory.forget Profinite).obj Z → (CategoryTheory.forget Profinit …
      hg' : Function.RightInverse g' ⇑g
      t : X → ↑Y.toTop := Function.comp g' (Function.comp (⇑f) Pure.pure)
      h : Ultrafilter X → ↑Y.toTop := Ultrafilter.extend t
      hh : Continuous h
      ⊢ Eq ((CategoryTheory.forget Profinite).map (CategoryTheory.CategoryStruct.com …
    -/
    convert denseRange_pure.equalizer (g.continuous.comp hh) f.continuous _
     -- Porting note: same fix as in `Topology.Category.CompHaus.Projective`
    /-
      case h.a
      X : Type u
      Y Z : Profinite
      f : Quiver.Hom (Profinite.of (Ultrafilter X)) Z
      g : Quiver.Hom Y Z
      hg : Function.Surjective ⇑g
      g' : (CategoryTheory.forget Profinite).obj Z → (CategoryTheory.forget Profinit …
      hg' : Function.RightInverse g' ⇑g
      t : X → ↑Y.toTop := Function.comp g' (Function.comp (⇑f) Pure.pure)
      h : Ultrafilter X → ↑Y.toTop := Ultrafilter.extend t
      hh : Continuous h
      ⊢ Eq (Function.comp (Function.comp (⇑g) h) Pure.pure) (Function.comp (⇑f) Pure …
    -/
    let g'' : ContinuousMap Y Z := g
    /-
      case h.a
      X : Type u
      Y Z : Profinite
      f : Quiver.Hom (Profinite.of (Ultrafilter X)) Z
      g : Quiver.Hom Y Z
      hg : Function.Surjective ⇑g
      g' : (CategoryTheory.forget Profinite).obj Z → (CategoryTheory.forget Profinit …
      hg' : Function.RightInverse g' ⇑g
      t : X → ↑Y.toTop := Function.comp g' (Function.comp (⇑f) Pure.pure)
      h : Ultrafilter X → ↑Y.toTop := Ultrafilter.extend t
      hh : Continuous h
      g'' : ContinuousMap ↑Y.toTop ↑Z.toTop := g
      ⊢ Eq (Function.comp (Function.comp (⇑g) h) Pure.pure) (Function.comp (⇑f) Pure …
    -/
    have : g'' ∘ g' = id := hg'.comp_eq_id
    -- This used to be `rw`, but we need `rw; rfl` after https://github.com/leanprover/lean4/pull/2644
    /-
      case h.a
      X : Type u
      Y Z : Profinite
      f : Quiver.Hom (Profinite.of (Ultrafilter X)) Z
      g : Quiver.Hom Y Z
      hg : Function.Surjective ⇑g
      g' : (CategoryTheory.forget Profinite).obj Z → (CategoryTheory.forget Profinit …
      hg' : Function.RightInverse g' ⇑g
      t : X → ↑Y.toTop := Function.comp g' (Function.comp (⇑f) Pure.pure)
      h : Ultrafilter X → ↑Y.toTop := Ultrafilter.extend t
      hh : Continuous h
      g'' : ContinuousMap ↑Y.toTop ↑Z.toTop := g
      this : Eq (Function.comp (⇑g'') g') id
      ⊢ Eq (Function.comp (Function.comp (⇑g) h) Pure.pure) (Function.comp (⇑f) Pure …
    -/
    rw [comp_assoc, ultrafilter_extend_extends, ← comp_assoc, this, id_comp]
    /-
      case h.a
      X : Type u
      Y Z : Profinite
      f : Quiver.Hom (Profinite.of (Ultrafilter X)) Z
      g : Quiver.Hom Y Z
      hg : Function.Surjective ⇑g
      g' : (CategoryTheory.forget Profinite).obj Z → (CategoryTheory.forget Profinit …
      hg' : Function.RightInverse g' ⇑g
      t : X → ↑Y.toTop := Function.comp g' (Function.comp (⇑f) Pure.pure)
      h : Ultrafilter X → ↑Y.toTop := Ultrafilter.extend t
      hh : Continuous h
      g'' : ContinuousMap ↑Y.toTop ↑Z.toTop := g
      this : Eq (Function.comp (⇑g'') g') id
      ⊢ Eq (Function.comp (⇑f) Pure.pure) (Function.comp (⇑f) Pure.pure)
    -/
    rfl
    /-
      🎉 no goals
    -/


/-- For any profinite `X`, the natural map `Ultrafilter X → X` is a projective presentation. -/
def projectivePresentation (X : Profinite.{u}) : ProjectivePresentation X where
  p := of <| Ultrafilter X
  f := ⟨_, continuous_ultrafilter_extend id⟩
  projective := Profinite.projective_ultrafilter X
  epi := ConcreteCategory.epi_of_surjective _ fun x =>
    ⟨(pure x : Ultrafilter X), congr_fun (ultrafilter_extend_extends (𝟙 X)) x⟩


instance : EnoughProjectives Profinite.{u} where presentation X := ⟨projectivePresentation X⟩


