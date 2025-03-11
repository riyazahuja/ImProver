lemma CommRingCat.epi_iff_tmul_eq_tmul {R S : Type u} [CommRing R] [CommRing S] [Algebra R S] :
    Epi (CommRingCat.ofHom (algebraMap R S)) ↔
      ∀ s : S, s ⊗ₜ[R] 1 = 1 ⊗ₜ s := by
  /-
    R S : Type u
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    ⊢ Iff (CategoryTheory.Epi (CommRingCat.ofHom (algebraMap R S))) (∀ (s : S), Eq …
  -/
  constructor
    /-
      case mp
      R S : Type u
      inst✝² : CommRing R
      inst✝¹ : CommRing S
      inst✝ : Algebra R S
      ⊢ CategoryTheory.Epi (CommRingCat.ofHom (algebraMap R S)) → ∀ (s : S), Eq (Ten …
    -/
  · intro H
    #adaptation_note
    /-- After https://github.com/leanprover/lean4/pull/6024
    we need to add `(R := R) (A := S)` in the next line to deal with unification issues. -/
    have := H.1 (CommRingCat.ofHom <| Algebra.TensorProduct.includeLeftRingHom (R := R))
      (CommRingCat.ofHom <| (Algebra.TensorProduct.includeRight (R := R) (A := S)).toRingHom)
      (by ext r; show algebraMap R S r ⊗ₜ 1 = 1 ⊗ₜ algebraMap R S r;
          simp only [Algebra.algebraMap_eq_smul_one, smul_tmul])
    /-
      case mp
      R S : Type u
      inst✝² : CommRing R
      inst✝¹ : CommRing S
      inst✝ : Algebra R S
      H : CategoryTheory.Epi (CommRingCat.ofHom (algebraMap R S))
      this : Eq (CommRingCat.ofHom Algebra.TensorProduct.includeLeftRingHom) (CommRi …
      ⊢ ∀ (s : S), Eq (TensorProduct.tmul R s 1) (TensorProduct.tmul R 1 s)
    -/
    exact RingHom.congr_fun (congrArg Hom.hom this)
    /-
      🎉 no goals
    -/
    /-
      case mpr
      R S : Type u
      inst✝² : CommRing R
      inst✝¹ : CommRing S
      inst✝ : Algebra R S
      ⊢ (∀ (s : S), Eq (TensorProduct.tmul R s 1) (TensorProduct.tmul R 1 s)) → Cate …
    -/
  · refine fun H ↦ ⟨fun {T} f g e ↦ ?_⟩
    /-
      case mpr
      R S : Type u
      inst✝² : CommRing R
      inst✝¹ : CommRing S
      inst✝ : Algebra R S
      H : ∀ (s : S), Eq (TensorProduct.tmul R s 1) (TensorProduct.tmul R 1 s)
      T : CommRingCat
      f g : Quiver.Hom (CommRingCat.of S) T
      e : Eq (CategoryTheory.CategoryStruct.comp (CommRingCat.ofHom (algebraMap R S) …
      ⊢ Eq f g
    -/
    letI : Algebra R T := (ofHom (algebraMap R S) ≫ g).hom.toAlgebra
    /-
      case mpr
      R S : Type u
      inst✝² : CommRing R
      inst✝¹ : CommRing S
      inst✝ : Algebra R S
      H : ∀ (s : S), Eq (TensorProduct.tmul R s 1) (TensorProduct.tmul R 1 s)
      T : CommRingCat
      f g : Quiver.Hom (CommRingCat.of S) T
      e : Eq (CategoryTheory.CategoryStruct.comp (CommRingCat.ofHom (algebraMap R S) …
      this : Algebra R ↑T := (CategoryTheory.CategoryStruct.comp (CommRingCat.ofHom  …
      ⊢ Eq f g
    -/
    let f' : S →ₐ[R] T := ⟨f.hom, RingHom.congr_fun (congrArg Hom.hom e)⟩
    /-
      case mpr
      R S : Type u
      inst✝² : CommRing R
      inst✝¹ : CommRing S
      inst✝ : Algebra R S
      H : ∀ (s : S), Eq (TensorProduct.tmul R s 1) (TensorProduct.tmul R 1 s)
      T : CommRingCat
      f g : Quiver.Hom (CommRingCat.of S) T
      e : Eq (CategoryTheory.CategoryStruct.comp (CommRingCat.ofHom (algebraMap R S) …
      this : Algebra R ↑T := (CategoryTheory.CategoryStruct.comp (CommRingCat.ofHom  …
      f' : AlgHom R S ↑T := { toRingHom := f.hom, commutes' := ⋯ }
      ⊢ Eq f g
    -/
    let g' : S →ₐ[R] T := ⟨g.hom, fun _ ↦ rfl⟩
    /-
      case mpr
      R S : Type u
      inst✝² : CommRing R
      inst✝¹ : CommRing S
      inst✝ : Algebra R S
      H : ∀ (s : S), Eq (TensorProduct.tmul R s 1) (TensorProduct.tmul R 1 s)
      T : CommRingCat
      f g : Quiver.Hom (CommRingCat.of S) T
      e : Eq (CategoryTheory.CategoryStruct.comp (CommRingCat.ofHom (algebraMap R S) …
      this : Algebra R ↑T := (CategoryTheory.CategoryStruct.comp (CommRingCat.ofHom  …
      f' : AlgHom R S ↑T := { toRingHom := f.hom, commutes' := ⋯ }
      g' : AlgHom R S ↑T := { toRingHom := g.hom, commutes' := ⋯ }
      ⊢ Eq f g
    -/
    ext s
    /-
      case mpr.hf.a
      R S : Type u
      inst✝² : CommRing R
      inst✝¹ : CommRing S
      inst✝ : Algebra R S
      H : ∀ (s : S), Eq (TensorProduct.tmul R s 1) (TensorProduct.tmul R 1 s)
      T : CommRingCat
      f g : Quiver.Hom (CommRingCat.of S) T
      e : Eq (CategoryTheory.CategoryStruct.comp (CommRingCat.ofHom (algebraMap R S) …
      this : Algebra R ↑T := (CategoryTheory.CategoryStruct.comp (CommRingCat.ofHom  …
      f' : AlgHom R S ↑T := { toRingHom := f.hom, commutes' := ⋯ }
      g' : AlgHom R S ↑T := { toRingHom := g.hom, commutes' := ⋯ }
      s : ↑(CommRingCat.of S)
      ⊢ Eq (f.hom s) (g.hom s)
    -/
    simpa using congr(Algebra.TensorProduct.lift f' g' (fun _ _ ↦ .all _ _) $(H s))
    /-
      🎉 no goals
    -/


lemma RingHom.surjective_of_epi_of_finite {R S : CommRingCat} (f : R ⟶ S) [Epi f]
    (h₂ : RingHom.Finite f.hom) : Function.Surjective f := by
  /-
    R S : CommRingCat
    f : Quiver.Hom R S
    inst✝ : CategoryTheory.Epi f
    h₂ : f.hom.Finite
    ⊢ Function.Surjective ⇑f.hom
  -/
  algebraize [f.hom]
  /-
    R S : CommRingCat
    f : Quiver.Hom R S
    inst✝ : CategoryTheory.Epi f
    h₂ : f.hom.Finite
    algInst✝ : Algebra ↑R ↑S := f.hom.toAlgebra
    algebraizeInst✝ : Module.Finite ↑R ↑S
    ⊢ Function.Surjective ⇑f.hom
  -/
  apply RingHom.surjective_of_tmul_eq_tmul_of_finite
  /-
    case h₁
    R S : CommRingCat
    f : Quiver.Hom R S
    inst✝ : CategoryTheory.Epi f
    h₂ : f.hom.Finite
    algInst✝ : Algebra ↑R ↑S := f.hom.toAlgebra
    algebraizeInst✝ : Module.Finite ↑R ↑S
    ⊢ ∀ (s : ↑S), Eq (TensorProduct.tmul (↑R) s 1) (TensorProduct.tmul (↑R) 1 s)
  -/
  rwa [← CommRingCat.epi_iff_tmul_eq_tmul]
  /-
    🎉 no goals
  -/


lemma RingHom.surjective_iff_epi_and_finite {R S : CommRingCat} {f : R ⟶ S} :
    Function.Surjective f ↔ Epi f ∧ RingHom.Finite f.hom where
  mp h := ⟨ConcreteCategory.epi_of_surjective f h, .of_surjective f.hom h⟩
  mpr := fun ⟨_, h⟩ ↦ surjective_of_epi_of_finite f h

