/--
To show `P` is a sheaf for the binding of `U` with `B`, it suffices to show that `P` is a sheaf for
`U`, that `P` is a sheaf for each sieve in `B`, and that it is separated for any pullback of any
sieve in `B`.

This is mostly an auxiliary lemma to show `isSheafFor_trans`.
Adapted from [Elephant], Lemma C2.1.7(i) with suggestions as mentioned in
https://math.stackexchange.com/a/358709/
-/
theorem isSheafFor_bind (P : Cᵒᵖ ⥤ Type v) (U : Sieve X) (B : ∀ ⦃Y⦄ ⦃f : Y ⟶ X⦄, U f → Sieve Y)
    (hU : Presieve.IsSheafFor P (U : Presieve X))
    (hB : ∀ ⦃Y⦄ ⦃f : Y ⟶ X⦄ (hf : U f), Presieve.IsSheafFor P (B hf : Presieve Y))
    (hB' : ∀ ⦃Y⦄ ⦃f : Y ⟶ X⦄ (h : U f) ⦃Z⦄ (g : Z ⟶ Y),
      Presieve.IsSeparatedFor P (((B h).pullback g) : Presieve Z)) :
    Presieve.IsSheafFor P (Sieve.bind (U : Presieve X) B : Presieve X) := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X : C
    P : CategoryTheory.Functor (Opposite C) (Type v)
    U : CategoryTheory.Sieve X
    B : ⦃Y : C⦄ → ⦃f : Quiver.Hom Y X⦄ → U.arrows f → CategoryTheory.Sieve Y
    hU : CategoryTheory.Presieve.IsSheafFor P U.arrows
    hB : ∀ ⦃Y : C⦄ ⦃f : Quiver.Hom Y X⦄ (hf : U.arrows f), CategoryTheory.Presieve …
    hB' : ∀ ⦃Y : C⦄ ⦃f : Quiver.Hom Y X⦄ (h : U.arrows f) ⦃Z : C⦄ (g : Quiver.Hom  …
    ⊢ CategoryTheory.Presieve.IsSheafFor P (CategoryTheory.Sieve.bind U.arrows B). …
  -/
  intro s hs
  let y : ∀ ⦃Y⦄ ⦃f : Y ⟶ X⦄ (hf : U f), Presieve.FamilyOfElements P (B hf : Presieve Y) :=
    fun Y f hf Z g hg => s _ (Presieve.bind_comp _ _ hg)
  have hy : ∀ ⦃Y⦄ ⦃f : Y ⟶ X⦄ (hf : U f), (y hf).Compatible := by
    intro Y f H Y₁ Y₂ Z g₁ g₂ f₁ f₂ hf₁ hf₂ comm
    apply hs
    apply reassoc_of% comm
  let t : Presieve.FamilyOfElements P (U : Presieve X) :=
    fun Y f hf => (hB hf).amalgamate (y hf) (hy hf)
  have ht : ∀ ⦃Y⦄ ⦃f : Y ⟶ X⦄ (hf : U f), (y hf).IsAmalgamation (t f hf) := fun Y f hf =>
    (hB hf).isAmalgamation _
  have hT : t.Compatible := by
    rw [Presieve.compatible_iff_sieveCompatible]
    intro Z W f h hf
    apply (hB (U.downward_closed hf h)).isSeparatedFor.ext
    intro Y l hl
    apply (hB' hf (l ≫ h)).ext
    intro M m hm
    have : bind U B (m ≫ l ≫ h ≫ f) := by simpa using (Presieve.bind_comp f hf hm : bind U B _)
    trans s (m ≫ l ≫ h ≫ f) this
    · have := ht (U.downward_closed hf h) _ ((B _).downward_closed hl m)
      rw [op_comp, FunctorToTypes.map_comp_apply] at this
      rw [this]
      change s _ _ = s _ _
      -- Porting note: the proof was `by simp`
      congr 1
      simp only [assoc]
    · have h : s _ _ = _ := (ht hf _ hm).symm
      -- Porting note: this was done by `simp only [assoc] at`
      conv_lhs at h => congr; rw [assoc, assoc]
      rw [h]
      simp only [op_comp, assoc, FunctorToTypes.map_comp_apply]
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X : C
    P : CategoryTheory.Functor (Opposite C) (Type v)
    U : CategoryTheory.Sieve X
    B : ⦃Y : C⦄ → ⦃f : Quiver.Hom Y X⦄ → U.arrows f → CategoryTheory.Sieve Y
    hU : CategoryTheory.Presieve.IsSheafFor P U.arrows
    hB : ∀ ⦃Y : C⦄ ⦃f : Quiver.Hom Y X⦄ (hf : U.arrows f), CategoryTheory.Presieve …
    hB' : ∀ ⦃Y : C⦄ ⦃f : Quiver.Hom Y X⦄ (h : U.arrows f) ⦃Z : C⦄ (g : Quiver.Hom  …
    s : CategoryTheory.Presieve.FamilyOfElements P (CategoryTheory.Sieve.bind U.ar …
    hs : s.Compatible
    y : ⦃Y : C⦄ → ⦃f : Quiver.Hom Y X⦄ → (hf : U.arrows f) → CategoryTheory.Presie …
    hy : ∀ ⦃Y : C⦄ ⦃f : Quiver.Hom Y X⦄ (hf : U.arrows f), (y hf).Compatible
    t : CategoryTheory.Presieve.FamilyOfElements P U.arrows := fun Y f hf => ⋯.ama …
    ht : ∀ ⦃Y : C⦄ ⦃f : Quiver.Hom Y X⦄ (hf : U.arrows f), (y hf).IsAmalgamation ( …
    hT : t.Compatible
    ⊢ ExistsUnique fun t => s.IsAmalgamation t
  -/
  refine ⟨hU.amalgamate t hT, ?_, ?_⟩
    /-
      case refine_1
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X : C
      P : CategoryTheory.Functor (Opposite C) (Type v)
      U : CategoryTheory.Sieve X
      B : ⦃Y : C⦄ → ⦃f : Quiver.Hom Y X⦄ → U.arrows f → CategoryTheory.Sieve Y
      hU : CategoryTheory.Presieve.IsSheafFor P U.arrows
      hB : ∀ ⦃Y : C⦄ ⦃f : Quiver.Hom Y X⦄ (hf : U.arrows f), CategoryTheory.Presieve …
      hB' : ∀ ⦃Y : C⦄ ⦃f : Quiver.Hom Y X⦄ (h : U.arrows f) ⦃Z : C⦄ (g : Quiver.Hom  …
      s : CategoryTheory.Presieve.FamilyOfElements P (CategoryTheory.Sieve.bind U.ar …
      hs : s.Compatible
      y : ⦃Y : C⦄ → ⦃f : Quiver.Hom Y X⦄ → (hf : U.arrows f) → CategoryTheory.Presie …
      hy : ∀ ⦃Y : C⦄ ⦃f : Quiver.Hom Y X⦄ (hf : U.arrows f), (y hf).Compatible
      t : CategoryTheory.Presieve.FamilyOfElements P U.arrows := fun Y f hf => ⋯.ama …
      ht : ∀ ⦃Y : C⦄ ⦃f : Quiver.Hom Y X⦄ (hf : U.arrows f), (y hf).IsAmalgamation ( …
      hT : t.Compatible
      ⊢ (fun t => s.IsAmalgamation t) (hU.amalgamate t hT)
    -/
  · rintro Z _ ⟨Y, f, g, hg, hf, rfl⟩
    /-
      case refine_1.intro.intro.intro.intro.intro
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X : C
      P : CategoryTheory.Functor (Opposite C) (Type v)
      U : CategoryTheory.Sieve X
      B : ⦃Y : C⦄ → ⦃f : Quiver.Hom Y X⦄ → U.arrows f → CategoryTheory.Sieve Y
      hU : CategoryTheory.Presieve.IsSheafFor P U.arrows
      hB : ∀ ⦃Y : C⦄ ⦃f : Quiver.Hom Y X⦄ (hf : U.arrows f), CategoryTheory.Presieve …
      hB' : ∀ ⦃Y : C⦄ ⦃f : Quiver.Hom Y X⦄ (h : U.arrows f) ⦃Z : C⦄ (g : Quiver.Hom  …
      s : CategoryTheory.Presieve.FamilyOfElements P (CategoryTheory.Sieve.bind U.ar …
      hs : s.Compatible
      y : ⦃Y : C⦄ → ⦃f : Quiver.Hom Y X⦄ → (hf : U.arrows f) → CategoryTheory.Presie …
      hy : ∀ ⦃Y : C⦄ ⦃f : Quiver.Hom Y X⦄ (hf : U.arrows f), (y hf).Compatible
      t : CategoryTheory.Presieve.FamilyOfElements P U.arrows := fun Y f hf => ⋯.ama …
      ht : ∀ ⦃Y : C⦄ ⦃f : Quiver.Hom Y X⦄ (hf : U.arrows f), (y hf).IsAmalgamation ( …
      hT : t.Compatible
      Z Y : C
      f : Quiver.Hom Z Y
      g : Quiver.Hom Y X
      hg : U.arrows g
      hf : (B hg).arrows f
      ⊢ Eq (P.map (CategoryTheory.CategoryStruct.comp f g).op (hU.amalgamate t hT))  …
    -/
    rw [op_comp, FunctorToTypes.map_comp_apply, Presieve.IsSheafFor.valid_glue _ _ _ hg]
    /-
      case refine_1.intro.intro.intro.intro.intro
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X : C
      P : CategoryTheory.Functor (Opposite C) (Type v)
      U : CategoryTheory.Sieve X
      B : ⦃Y : C⦄ → ⦃f : Quiver.Hom Y X⦄ → U.arrows f → CategoryTheory.Sieve Y
      hU : CategoryTheory.Presieve.IsSheafFor P U.arrows
      hB : ∀ ⦃Y : C⦄ ⦃f : Quiver.Hom Y X⦄ (hf : U.arrows f), CategoryTheory.Presieve …
      hB' : ∀ ⦃Y : C⦄ ⦃f : Quiver.Hom Y X⦄ (h : U.arrows f) ⦃Z : C⦄ (g : Quiver.Hom  …
      s : CategoryTheory.Presieve.FamilyOfElements P (CategoryTheory.Sieve.bind U.ar …
      hs : s.Compatible
      y : ⦃Y : C⦄ → ⦃f : Quiver.Hom Y X⦄ → (hf : U.arrows f) → CategoryTheory.Presie …
      hy : ∀ ⦃Y : C⦄ ⦃f : Quiver.Hom Y X⦄ (hf : U.arrows f), (y hf).Compatible
      t : CategoryTheory.Presieve.FamilyOfElements P U.arrows := fun Y f hf => ⋯.ama …
      ht : ∀ ⦃Y : C⦄ ⦃f : Quiver.Hom Y X⦄ (hf : U.arrows f), (y hf).IsAmalgamation ( …
      hT : t.Compatible
      Z Y : C
      f : Quiver.Hom Z Y
      g : Quiver.Hom Y X
      hg : U.arrows g
      hf : (B hg).arrows f
      ⊢ Eq (P.map f.op (t g hg)) (s (CategoryTheory.CategoryStruct.comp f g) ⋯)
    -/
    apply ht hg _ hf
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X : C
      P : CategoryTheory.Functor (Opposite C) (Type v)
      U : CategoryTheory.Sieve X
      B : ⦃Y : C⦄ → ⦃f : Quiver.Hom Y X⦄ → U.arrows f → CategoryTheory.Sieve Y
      hU : CategoryTheory.Presieve.IsSheafFor P U.arrows
      hB : ∀ ⦃Y : C⦄ ⦃f : Quiver.Hom Y X⦄ (hf : U.arrows f), CategoryTheory.Presieve …
      hB' : ∀ ⦃Y : C⦄ ⦃f : Quiver.Hom Y X⦄ (h : U.arrows f) ⦃Z : C⦄ (g : Quiver.Hom  …
      s : CategoryTheory.Presieve.FamilyOfElements P (CategoryTheory.Sieve.bind U.ar …
      hs : s.Compatible
      y : ⦃Y : C⦄ → ⦃f : Quiver.Hom Y X⦄ → (hf : U.arrows f) → CategoryTheory.Presie …
      hy : ∀ ⦃Y : C⦄ ⦃f : Quiver.Hom Y X⦄ (hf : U.arrows f), (y hf).Compatible
      t : CategoryTheory.Presieve.FamilyOfElements P U.arrows := fun Y f hf => ⋯.ama …
      ht : ∀ ⦃Y : C⦄ ⦃f : Quiver.Hom Y X⦄ (hf : U.arrows f), (y hf).IsAmalgamation ( …
      hT : t.Compatible
      ⊢ ∀ (y : P.obj { unop := X }), (fun t => s.IsAmalgamation t) y → Eq y (hU.amal …
    -/
  · intro y hy
    /-
      case refine_2
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X : C
      P : CategoryTheory.Functor (Opposite C) (Type v)
      U : CategoryTheory.Sieve X
      B : ⦃Y : C⦄ → ⦃f : Quiver.Hom Y X⦄ → U.arrows f → CategoryTheory.Sieve Y
      hU : CategoryTheory.Presieve.IsSheafFor P U.arrows
      hB : ∀ ⦃Y : C⦄ ⦃f : Quiver.Hom Y X⦄ (hf : U.arrows f), CategoryTheory.Presieve …
      hB' : ∀ ⦃Y : C⦄ ⦃f : Quiver.Hom Y X⦄ (h : U.arrows f) ⦃Z : C⦄ (g : Quiver.Hom  …
      s : CategoryTheory.Presieve.FamilyOfElements P (CategoryTheory.Sieve.bind U.ar …
      hs : s.Compatible
      y✝ : ⦃Y : C⦄ → ⦃f : Quiver.Hom Y X⦄ → (hf : U.arrows f) → CategoryTheory.Presi …
      hy✝ : ∀ ⦃Y : C⦄ ⦃f : Quiver.Hom Y X⦄ (hf : U.arrows f), (y✝ hf).Compatible
      t : CategoryTheory.Presieve.FamilyOfElements P U.arrows := fun Y f hf => ⋯.ama …
      ht : ∀ ⦃Y : C⦄ ⦃f : Quiver.Hom Y X⦄ (hf : U.arrows f), (y✝ hf).IsAmalgamation  …
      hT : t.Compatible
      y : P.obj { unop := X }
      hy : s.IsAmalgamation y
      ⊢ Eq y (hU.amalgamate t hT)
    -/
    apply hU.isSeparatedFor.ext
    /-
      case refine_2
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X : C
      P : CategoryTheory.Functor (Opposite C) (Type v)
      U : CategoryTheory.Sieve X
      B : ⦃Y : C⦄ → ⦃f : Quiver.Hom Y X⦄ → U.arrows f → CategoryTheory.Sieve Y
      hU : CategoryTheory.Presieve.IsSheafFor P U.arrows
      hB : ∀ ⦃Y : C⦄ ⦃f : Quiver.Hom Y X⦄ (hf : U.arrows f), CategoryTheory.Presieve …
      hB' : ∀ ⦃Y : C⦄ ⦃f : Quiver.Hom Y X⦄ (h : U.arrows f) ⦃Z : C⦄ (g : Quiver.Hom  …
      s : CategoryTheory.Presieve.FamilyOfElements P (CategoryTheory.Sieve.bind U.ar …
      hs : s.Compatible
      y✝ : ⦃Y : C⦄ → ⦃f : Quiver.Hom Y X⦄ → (hf : U.arrows f) → CategoryTheory.Presi …
      hy✝ : ∀ ⦃Y : C⦄ ⦃f : Quiver.Hom Y X⦄ (hf : U.arrows f), (y✝ hf).Compatible
      t : CategoryTheory.Presieve.FamilyOfElements P U.arrows := fun Y f hf => ⋯.ama …
      ht : ∀ ⦃Y : C⦄ ⦃f : Quiver.Hom Y X⦄ (hf : U.arrows f), (y✝ hf).IsAmalgamation  …
      hT : t.Compatible
      y : P.obj { unop := X }
      hy : s.IsAmalgamation y
      ⊢ ∀ ⦃Y : C⦄ ⦃f : Quiver.Hom Y X⦄, U.arrows f → Eq (P.map f.op y) (P.map f.op ( …
    -/
    intro Y f hf
    /-
      case refine_2
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X : C
      P : CategoryTheory.Functor (Opposite C) (Type v)
      U : CategoryTheory.Sieve X
      B : ⦃Y : C⦄ → ⦃f : Quiver.Hom Y X⦄ → U.arrows f → CategoryTheory.Sieve Y
      hU : CategoryTheory.Presieve.IsSheafFor P U.arrows
      hB : ∀ ⦃Y : C⦄ ⦃f : Quiver.Hom Y X⦄ (hf : U.arrows f), CategoryTheory.Presieve …
      hB' : ∀ ⦃Y : C⦄ ⦃f : Quiver.Hom Y X⦄ (h : U.arrows f) ⦃Z : C⦄ (g : Quiver.Hom  …
      s : CategoryTheory.Presieve.FamilyOfElements P (CategoryTheory.Sieve.bind U.ar …
      hs : s.Compatible
      y✝ : ⦃Y : C⦄ → ⦃f : Quiver.Hom Y X⦄ → (hf : U.arrows f) → CategoryTheory.Presi …
      hy✝ : ∀ ⦃Y : C⦄ ⦃f : Quiver.Hom Y X⦄ (hf : U.arrows f), (y✝ hf).Compatible
      t : CategoryTheory.Presieve.FamilyOfElements P U.arrows := fun Y f hf => ⋯.ama …
      ht : ∀ ⦃Y : C⦄ ⦃f : Quiver.Hom Y X⦄ (hf : U.arrows f), (y✝ hf).IsAmalgamation  …
      hT : t.Compatible
      y : P.obj { unop := X }
      hy : s.IsAmalgamation y
      Y : C
      f : Quiver.Hom Y X
      hf : U.arrows f
      ⊢ Eq (P.map f.op y) (P.map f.op (hU.amalgamate t hT))
    -/
    apply (hB hf).isSeparatedFor.ext
    /-
      case refine_2
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X : C
      P : CategoryTheory.Functor (Opposite C) (Type v)
      U : CategoryTheory.Sieve X
      B : ⦃Y : C⦄ → ⦃f : Quiver.Hom Y X⦄ → U.arrows f → CategoryTheory.Sieve Y
      hU : CategoryTheory.Presieve.IsSheafFor P U.arrows
      hB : ∀ ⦃Y : C⦄ ⦃f : Quiver.Hom Y X⦄ (hf : U.arrows f), CategoryTheory.Presieve …
      hB' : ∀ ⦃Y : C⦄ ⦃f : Quiver.Hom Y X⦄ (h : U.arrows f) ⦃Z : C⦄ (g : Quiver.Hom  …
      s : CategoryTheory.Presieve.FamilyOfElements P (CategoryTheory.Sieve.bind U.ar …
      hs : s.Compatible
      y✝ : ⦃Y : C⦄ → ⦃f : Quiver.Hom Y X⦄ → (hf : U.arrows f) → CategoryTheory.Presi …
      hy✝ : ∀ ⦃Y : C⦄ ⦃f : Quiver.Hom Y X⦄ (hf : U.arrows f), (y✝ hf).Compatible
      t : CategoryTheory.Presieve.FamilyOfElements P U.arrows := fun Y f hf => ⋯.ama …
      ht : ∀ ⦃Y : C⦄ ⦃f : Quiver.Hom Y X⦄ (hf : U.arrows f), (y✝ hf).IsAmalgamation  …
      hT : t.Compatible
      y : P.obj { unop := X }
      hy : s.IsAmalgamation y
      Y : C
      f : Quiver.Hom Y X
      hf : U.arrows f
      ⊢ ∀ ⦃Y_1 : C⦄ ⦃f_1 : Quiver.Hom Y_1 Y⦄, (B hf).arrows f_1 → Eq (P.map f_1.op ( …
    -/
    intro Z g hg
    rw [← FunctorToTypes.map_comp_apply, ← op_comp, hy _ (Presieve.bind_comp _ _ hg),
      hU.valid_glue _ _ hf, ht hf _ hg]


/-- Given two sieves `R` and `S`, to show that `P` is a sheaf for `S`, we can show:
* `P` is a sheaf for `R`
* `P` is a sheaf for the pullback of `S` along any arrow in `R`
* `P` is separated for the pullback of `R` along any arrow in `S`.

This is mostly an auxiliary lemma to construct `finestTopology`.
Adapted from [Elephant], Lemma C2.1.7(ii) with suggestions as mentioned in
https://math.stackexchange.com/a/358709
-/
theorem isSheafFor_trans (P : Cᵒᵖ ⥤ Type v) (R S : Sieve X)
    (hR : Presieve.IsSheafFor P (R : Presieve X))
    (hR' : ∀ ⦃Y⦄ ⦃f : Y ⟶ X⦄ (_ : S f), Presieve.IsSeparatedFor P (R.pullback f : Presieve Y))
    (hS : ∀ ⦃Y⦄ ⦃f : Y ⟶ X⦄ (_ : R f), Presieve.IsSheafFor P (S.pullback f : Presieve Y)) :
    Presieve.IsSheafFor P (S : Presieve X) := by
  have : (bind R fun Y f _ => S.pullback f : Presieve X) ≤ S := by
    rintro Z f ⟨W, f, g, hg, hf : S _, rfl⟩
    apply hf
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X : C
    P : CategoryTheory.Functor (Opposite C) (Type v)
    R S : CategoryTheory.Sieve X
    hR : CategoryTheory.Presieve.IsSheafFor P R.arrows
    hR' : ∀ ⦃Y : C⦄ ⦃f : Quiver.Hom Y X⦄, S.arrows f → CategoryTheory.Presieve.IsS …
    hS : ∀ ⦃Y : C⦄ ⦃f : Quiver.Hom Y X⦄, R.arrows f → CategoryTheory.Presieve.IsSh …
    this : LE.le (CategoryTheory.Sieve.bind R.arrows fun Y f x => CategoryTheory.S …
    ⊢ CategoryTheory.Presieve.IsSheafFor P S.arrows
  -/
  apply Presieve.isSheafFor_subsieve_aux P this
    /-
      case hS
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X : C
      P : CategoryTheory.Functor (Opposite C) (Type v)
      R S : CategoryTheory.Sieve X
      hR : CategoryTheory.Presieve.IsSheafFor P R.arrows
      hR' : ∀ ⦃Y : C⦄ ⦃f : Quiver.Hom Y X⦄, S.arrows f → CategoryTheory.Presieve.IsS …
      hS : ∀ ⦃Y : C⦄ ⦃f : Quiver.Hom Y X⦄, R.arrows f → CategoryTheory.Presieve.IsSh …
      this : LE.le (CategoryTheory.Sieve.bind R.arrows fun Y f x => CategoryTheory.S …
      ⊢ CategoryTheory.Presieve.IsSheafFor P (CategoryTheory.Sieve.bind R.arrows fun …
    -/
  · apply isSheafFor_bind _ _ _ hR hS
    /-
      case hS
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X : C
      P : CategoryTheory.Functor (Opposite C) (Type v)
      R S : CategoryTheory.Sieve X
      hR : CategoryTheory.Presieve.IsSheafFor P R.arrows
      hR' : ∀ ⦃Y : C⦄ ⦃f : Quiver.Hom Y X⦄, S.arrows f → CategoryTheory.Presieve.IsS …
      hS : ∀ ⦃Y : C⦄ ⦃f : Quiver.Hom Y X⦄, R.arrows f → CategoryTheory.Presieve.IsSh …
      this : LE.le (CategoryTheory.Sieve.bind R.arrows fun Y f x => CategoryTheory.S …
      ⊢ ∀ ⦃Y : C⦄ ⦃f : Quiver.Hom Y X⦄, R.arrows f → ∀ ⦃Z : C⦄ (g : Quiver.Hom Z Y), …
    -/
    intro Y f hf Z g
    /-
      case hS
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X : C
      P : CategoryTheory.Functor (Opposite C) (Type v)
      R S : CategoryTheory.Sieve X
      hR : CategoryTheory.Presieve.IsSheafFor P R.arrows
      hR' : ∀ ⦃Y : C⦄ ⦃f : Quiver.Hom Y X⦄, S.arrows f → CategoryTheory.Presieve.IsS …
      hS : ∀ ⦃Y : C⦄ ⦃f : Quiver.Hom Y X⦄, R.arrows f → CategoryTheory.Presieve.IsSh …
      this : LE.le (CategoryTheory.Sieve.bind R.arrows fun Y f x => CategoryTheory.S …
      Y : C
      f : Quiver.Hom Y X
      hf : R.arrows f
      Z : C
      g : Quiver.Hom Z Y
      ⊢ CategoryTheory.Presieve.IsSeparatedFor P (CategoryTheory.Sieve.pullback g (C …
    -/
    rw [← pullback_comp]
    /-
      case hS
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X : C
      P : CategoryTheory.Functor (Opposite C) (Type v)
      R S : CategoryTheory.Sieve X
      hR : CategoryTheory.Presieve.IsSheafFor P R.arrows
      hR' : ∀ ⦃Y : C⦄ ⦃f : Quiver.Hom Y X⦄, S.arrows f → CategoryTheory.Presieve.IsS …
      hS : ∀ ⦃Y : C⦄ ⦃f : Quiver.Hom Y X⦄, R.arrows f → CategoryTheory.Presieve.IsSh …
      this : LE.le (CategoryTheory.Sieve.bind R.arrows fun Y f x => CategoryTheory.S …
      Y : C
      f : Quiver.Hom Y X
      hf : R.arrows f
      Z : C
      g : Quiver.Hom Z Y
      ⊢ CategoryTheory.Presieve.IsSeparatedFor P (CategoryTheory.Sieve.pullback (Cat …
    -/
    apply (hS (R.downward_closed hf _)).isSeparatedFor
    /-
      🎉 no goals
    -/
    /-
      case trans
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X : C
      P : CategoryTheory.Functor (Opposite C) (Type v)
      R S : CategoryTheory.Sieve X
      hR : CategoryTheory.Presieve.IsSheafFor P R.arrows
      hR' : ∀ ⦃Y : C⦄ ⦃f : Quiver.Hom Y X⦄, S.arrows f → CategoryTheory.Presieve.IsS …
      hS : ∀ ⦃Y : C⦄ ⦃f : Quiver.Hom Y X⦄, R.arrows f → CategoryTheory.Presieve.IsSh …
      this : LE.le (CategoryTheory.Sieve.bind R.arrows fun Y f x => CategoryTheory.S …
      ⊢ ∀ ⦃Y : C⦄ ⦃f : Quiver.Hom Y X⦄, S.arrows f → CategoryTheory.Presieve.IsSepar …
    -/
  · intro Y f hf
    have : Sieve.pullback f (bind R fun T (k : T ⟶ X) (_ : R k) => pullback k S) =
        R.pullback f := by
      ext Z g
      constructor
      · rintro ⟨W, k, l, hl, _, comm⟩
        rw [pullback_apply, ← comm]
        simp [hl]
      · intro a
        refine ⟨Z, 𝟙 Z, _, a, ?_⟩
        simp [hf]
    /-
      case trans
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X : C
      P : CategoryTheory.Functor (Opposite C) (Type v)
      R S : CategoryTheory.Sieve X
      hR : CategoryTheory.Presieve.IsSheafFor P R.arrows
      hR' : ∀ ⦃Y : C⦄ ⦃f : Quiver.Hom Y X⦄, S.arrows f → CategoryTheory.Presieve.IsS …
      hS : ∀ ⦃Y : C⦄ ⦃f : Quiver.Hom Y X⦄, R.arrows f → CategoryTheory.Presieve.IsSh …
      this✝ : LE.le (CategoryTheory.Sieve.bind R.arrows fun Y f x => CategoryTheory. …
      Y : C
      f : Quiver.Hom Y X
      hf : S.arrows f
      this : Eq (CategoryTheory.Sieve.pullback f (CategoryTheory.Sieve.bind R.arrows …
      ⊢ CategoryTheory.Presieve.IsSeparatedFor P (CategoryTheory.Sieve.pullback f (C …
    -/
    rw [this]
    /-
      case trans
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X : C
      P : CategoryTheory.Functor (Opposite C) (Type v)
      R S : CategoryTheory.Sieve X
      hR : CategoryTheory.Presieve.IsSheafFor P R.arrows
      hR' : ∀ ⦃Y : C⦄ ⦃f : Quiver.Hom Y X⦄, S.arrows f → CategoryTheory.Presieve.IsS …
      hS : ∀ ⦃Y : C⦄ ⦃f : Quiver.Hom Y X⦄, R.arrows f → CategoryTheory.Presieve.IsSh …
      this✝ : LE.le (CategoryTheory.Sieve.bind R.arrows fun Y f x => CategoryTheory. …
      Y : C
      f : Quiver.Hom Y X
      hf : S.arrows f
      this : Eq (CategoryTheory.Sieve.pullback f (CategoryTheory.Sieve.bind R.arrows …
      ⊢ CategoryTheory.Presieve.IsSeparatedFor P (CategoryTheory.Sieve.pullback f R) …
    -/
    apply hR' hf
    /-
      🎉 no goals
    -/


/-- Construct the finest (largest) Grothendieck topology for which the given presheaf is a sheaf.

This is a special case of https://stacks.math.columbia.edu/tag/00Z9, but following a different
proof (see the comments there).
-/
def finestTopologySingle (P : Cᵒᵖ ⥤ Type v) : GrothendieckTopology C where
  sieves X S := ∀ (Y) (f : Y ⟶ X), Presieve.IsSheafFor P (S.pullback f : Presieve Y)
  top_mem' X Y f := by
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      P✝ : CategoryTheory.Functor (Opposite C) (Type v)
      X✝ : C
      J : CategoryTheory.GrothendieckTopology C
      P : CategoryTheory.Functor (Opposite C) (Type v)
      X Y : C
      f : Quiver.Hom Y X
      ⊢ CategoryTheory.Presieve.IsSheafFor P (CategoryTheory.Sieve.pullback f Top.to …
    -/
    rw [Sieve.pullback_top]
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      P✝ : CategoryTheory.Functor (Opposite C) (Type v)
      X✝ : C
      J : CategoryTheory.GrothendieckTopology C
      P : CategoryTheory.Functor (Opposite C) (Type v)
      X Y : C
      f : Quiver.Hom Y X
      ⊢ CategoryTheory.Presieve.IsSheafFor P Top.top.arrows
    -/
    exact Presieve.isSheafFor_top_sieve P
    /-
      🎉 no goals
    -/
  pullback_stable' X Y S f hS Z g := by
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      P✝ : CategoryTheory.Functor (Opposite C) (Type v)
      X✝ : C
      J : CategoryTheory.GrothendieckTopology C
      P : CategoryTheory.Functor (Opposite C) (Type v)
      X Y : C
      S : CategoryTheory.Sieve X
      f : Quiver.Hom Y X
      hS : Membership.mem ((fun X S => ∀ (Y : C) (f : Quiver.Hom Y X), CategoryTheor …
      Z : C
      g : Quiver.Hom Z Y
      ⊢ CategoryTheory.Presieve.IsSheafFor P (CategoryTheory.Sieve.pullback g (Categ …
    -/
    rw [← pullback_comp]
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      P✝ : CategoryTheory.Functor (Opposite C) (Type v)
      X✝ : C
      J : CategoryTheory.GrothendieckTopology C
      P : CategoryTheory.Functor (Opposite C) (Type v)
      X Y : C
      S : CategoryTheory.Sieve X
      f : Quiver.Hom Y X
      hS : Membership.mem ((fun X S => ∀ (Y : C) (f : Quiver.Hom Y X), CategoryTheor …
      Z : C
      g : Quiver.Hom Z Y
      ⊢ CategoryTheory.Presieve.IsSheafFor P (CategoryTheory.Sieve.pullback (Categor …
    -/
    apply hS
    /-
      🎉 no goals
    -/
  transitive' X S hS R hR Z g := by
    -- This is the hard part of the construction, showing that the given set of sieves satisfies
    -- the transitivity axiom.
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      P✝ : CategoryTheory.Functor (Opposite C) (Type v)
      X✝ : C
      J : CategoryTheory.GrothendieckTopology C
      P : CategoryTheory.Functor (Opposite C) (Type v)
      X : C
      S : CategoryTheory.Sieve X
      hS : Membership.mem ((fun X S => ∀ (Y : C) (f : Quiver.Hom Y X), CategoryTheor …
      R : CategoryTheory.Sieve X
      hR : ∀ ⦃Y : C⦄ ⦃f : Quiver.Hom Y X⦄, S.arrows f → Membership.mem ((fun X S =>  …
      Z : C
      g : Quiver.Hom Z X
      ⊢ CategoryTheory.Presieve.IsSheafFor P (CategoryTheory.Sieve.pullback g R).arr …
    -/
    refine isSheafFor_trans P (pullback g S) _ (hS Z g) ?_ ?_
      /-
        case refine_1
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        P✝ : CategoryTheory.Functor (Opposite C) (Type v)
        X✝ : C
        J : CategoryTheory.GrothendieckTopology C
        P : CategoryTheory.Functor (Opposite C) (Type v)
        X : C
        S : CategoryTheory.Sieve X
        hS : Membership.mem ((fun X S => ∀ (Y : C) (f : Quiver.Hom Y X), CategoryTheor …
        R : CategoryTheory.Sieve X
        hR : ∀ ⦃Y : C⦄ ⦃f : Quiver.Hom Y X⦄, S.arrows f → Membership.mem ((fun X S =>  …
        Z : C
        g : Quiver.Hom Z X
        ⊢ ∀ ⦃Y : C⦄ ⦃f : Quiver.Hom Y Z⦄, (CategoryTheory.Sieve.pullback g R).arrows f …
      -/
    · intro Y f _
      /-
        case refine_1
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        P✝ : CategoryTheory.Functor (Opposite C) (Type v)
        X✝ : C
        J : CategoryTheory.GrothendieckTopology C
        P : CategoryTheory.Functor (Opposite C) (Type v)
        X : C
        S : CategoryTheory.Sieve X
        hS : Membership.mem ((fun X S => ∀ (Y : C) (f : Quiver.Hom Y X), CategoryTheor …
        R : CategoryTheory.Sieve X
        hR : ∀ ⦃Y : C⦄ ⦃f : Quiver.Hom Y X⦄, S.arrows f → Membership.mem ((fun X S =>  …
        Z : C
        g : Quiver.Hom Z X
        Y : C
        f : Quiver.Hom Y Z
        x✝ : (CategoryTheory.Sieve.pullback g R).arrows f
        ⊢ CategoryTheory.Presieve.IsSeparatedFor P (CategoryTheory.Sieve.pullback f (C …
      -/
      rw [← pullback_comp]
      /-
        case refine_1
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        P✝ : CategoryTheory.Functor (Opposite C) (Type v)
        X✝ : C
        J : CategoryTheory.GrothendieckTopology C
        P : CategoryTheory.Functor (Opposite C) (Type v)
        X : C
        S : CategoryTheory.Sieve X
        hS : Membership.mem ((fun X S => ∀ (Y : C) (f : Quiver.Hom Y X), CategoryTheor …
        R : CategoryTheory.Sieve X
        hR : ∀ ⦃Y : C⦄ ⦃f : Quiver.Hom Y X⦄, S.arrows f → Membership.mem ((fun X S =>  …
        Z : C
        g : Quiver.Hom Z X
        Y : C
        f : Quiver.Hom Y Z
        x✝ : (CategoryTheory.Sieve.pullback g R).arrows f
        ⊢ CategoryTheory.Presieve.IsSeparatedFor P (CategoryTheory.Sieve.pullback (Cat …
      -/
      apply (hS _ _).isSeparatedFor
      /-
        🎉 no goals
      -/
      /-
        case refine_2
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        P✝ : CategoryTheory.Functor (Opposite C) (Type v)
        X✝ : C
        J : CategoryTheory.GrothendieckTopology C
        P : CategoryTheory.Functor (Opposite C) (Type v)
        X : C
        S : CategoryTheory.Sieve X
        hS : Membership.mem ((fun X S => ∀ (Y : C) (f : Quiver.Hom Y X), CategoryTheor …
        R : CategoryTheory.Sieve X
        hR : ∀ ⦃Y : C⦄ ⦃f : Quiver.Hom Y X⦄, S.arrows f → Membership.mem ((fun X S =>  …
        Z : C
        g : Quiver.Hom Z X
        ⊢ ∀ ⦃Y : C⦄ ⦃f : Quiver.Hom Y Z⦄, (CategoryTheory.Sieve.pullback g S).arrows f …
      -/
    · intro Y f hf
      /-
        case refine_2
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        P✝ : CategoryTheory.Functor (Opposite C) (Type v)
        X✝ : C
        J : CategoryTheory.GrothendieckTopology C
        P : CategoryTheory.Functor (Opposite C) (Type v)
        X : C
        S : CategoryTheory.Sieve X
        hS : Membership.mem ((fun X S => ∀ (Y : C) (f : Quiver.Hom Y X), CategoryTheor …
        R : CategoryTheory.Sieve X
        hR : ∀ ⦃Y : C⦄ ⦃f : Quiver.Hom Y X⦄, S.arrows f → Membership.mem ((fun X S =>  …
        Z : C
        g : Quiver.Hom Z X
        Y : C
        f : Quiver.Hom Y Z
        hf : (CategoryTheory.Sieve.pullback g S).arrows f
        ⊢ CategoryTheory.Presieve.IsSheafFor P (CategoryTheory.Sieve.pullback f (Categ …
      -/
      have := hR hf _ (𝟙 _)
      /-
        case refine_2
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        P✝ : CategoryTheory.Functor (Opposite C) (Type v)
        X✝ : C
        J : CategoryTheory.GrothendieckTopology C
        P : CategoryTheory.Functor (Opposite C) (Type v)
        X : C
        S : CategoryTheory.Sieve X
        hS : Membership.mem ((fun X S => ∀ (Y : C) (f : Quiver.Hom Y X), CategoryTheor …
        R : CategoryTheory.Sieve X
        hR : ∀ ⦃Y : C⦄ ⦃f : Quiver.Hom Y X⦄, S.arrows f → Membership.mem ((fun X S =>  …
        Z : C
        g : Quiver.Hom Z X
        Y : C
        f : Quiver.Hom Y Z
        hf : (CategoryTheory.Sieve.pullback g S).arrows f
        this : CategoryTheory.Presieve.IsSheafFor P (CategoryTheory.Sieve.pullback (Ca …
        ⊢ CategoryTheory.Presieve.IsSheafFor P (CategoryTheory.Sieve.pullback f (Categ …
      -/
      rw [pullback_id, pullback_comp] at this
      /-
        case refine_2
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        P✝ : CategoryTheory.Functor (Opposite C) (Type v)
        X✝ : C
        J : CategoryTheory.GrothendieckTopology C
        P : CategoryTheory.Functor (Opposite C) (Type v)
        X : C
        S : CategoryTheory.Sieve X
        hS : Membership.mem ((fun X S => ∀ (Y : C) (f : Quiver.Hom Y X), CategoryTheor …
        R : CategoryTheory.Sieve X
        hR : ∀ ⦃Y : C⦄ ⦃f : Quiver.Hom Y X⦄, S.arrows f → Membership.mem ((fun X S =>  …
        Z : C
        g : Quiver.Hom Z X
        Y : C
        f : Quiver.Hom Y Z
        hf : (CategoryTheory.Sieve.pullback g S).arrows f
        this : CategoryTheory.Presieve.IsSheafFor P (CategoryTheory.Sieve.pullback f ( …
        ⊢ CategoryTheory.Presieve.IsSheafFor P (CategoryTheory.Sieve.pullback f (Categ …
      -/
      apply this
      /-
        🎉 no goals
      -/


/--
Construct the finest (largest) Grothendieck topology for which all the given presheaves are sheaves.

This is equal to the construction of <https://stacks.math.columbia.edu/tag/00Z9>.
-/
def finestTopology (Ps : Set (Cᵒᵖ ⥤ Type v)) : GrothendieckTopology C :=
  sInf (finestTopologySingle '' Ps)


/-- Check that if `P ∈ Ps`, then `P` is indeed a sheaf for the finest topology on `Ps`. -/
theorem sheaf_for_finestTopology (Ps : Set (Cᵒᵖ ⥤ Type v)) (h : P ∈ Ps) :
    Presieve.IsSheaf (finestTopology Ps) P := fun X S hS => by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    P : CategoryTheory.Functor (Opposite C) (Type v)
    Ps : Set (CategoryTheory.Functor (Opposite C) (Type v))
    h : Membership.mem Ps P
    X : C
    S : CategoryTheory.Sieve X
    hS : Membership.mem ((CategoryTheory.Sheaf.finestTopology Ps) X) S
    ⊢ CategoryTheory.Presieve.IsSheafFor P S.arrows
  -/
  simpa using hS _ ⟨⟨_, _, ⟨_, h, rfl⟩, rfl⟩, rfl⟩ _ (𝟙 _)
  /-
    🎉 no goals
  -/


/--
Check that if each `P ∈ Ps` is a sheaf for `J`, then `J` is a subtopology of `finestTopology Ps`.
-/
theorem le_finestTopology (Ps : Set (Cᵒᵖ ⥤ Type v)) (J : GrothendieckTopology C)
    (hJ : ∀ P ∈ Ps, Presieve.IsSheaf J P) : J ≤ finestTopology Ps := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    Ps : Set (CategoryTheory.Functor (Opposite C) (Type v))
    J : CategoryTheory.GrothendieckTopology C
    hJ : ∀ (P : CategoryTheory.Functor (Opposite C) (Type v)), Membership.mem Ps P …
    ⊢ LE.le J (CategoryTheory.Sheaf.finestTopology Ps)
  -/
  rintro X S hS _ ⟨⟨_, _, ⟨P, hP, rfl⟩, rfl⟩, rfl⟩
  /-
    case intro.mk.intro.intro.intro.intro
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    Ps : Set (CategoryTheory.Functor (Opposite C) (Type v))
    J : CategoryTheory.GrothendieckTopology C
    hJ : ∀ (P : CategoryTheory.Functor (Opposite C) (Type v)), Membership.mem Ps P …
    X : C
    S : CategoryTheory.Sieve X
    hS : Membership.mem (J X) S
    P : CategoryTheory.Functor (Opposite C) (Type v)
    hP : Membership.mem Ps P
    ⊢ Membership.mem ((fun f => ↑f X) ⟨(CategoryTheory.Sheaf.finestTopologySingle  …
  -/
  intro Y f
  -- this can't be combined with the previous because the `subst` is applied at the end
  /-
    case intro.mk.intro.intro.intro.intro
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    Ps : Set (CategoryTheory.Functor (Opposite C) (Type v))
    J : CategoryTheory.GrothendieckTopology C
    hJ : ∀ (P : CategoryTheory.Functor (Opposite C) (Type v)), Membership.mem Ps P …
    X : C
    S : CategoryTheory.Sieve X
    hS : Membership.mem (J X) S
    P : CategoryTheory.Functor (Opposite C) (Type v)
    hP : Membership.mem Ps P
    Y : C
    f : Quiver.Hom Y X
    ⊢ CategoryTheory.Presieve.IsSheafFor P (CategoryTheory.Sieve.pullback f S).arr …
  -/
  exact hJ P hP (S.pullback f) (J.pullback_stable f hS)
  /-
    🎉 no goals
  -/


/-- The `canonicalTopology` on a category is the finest (largest) topology for which every
representable presheaf is a sheaf.

See <https://stacks.math.columbia.edu/tag/00ZA>
-/
def canonicalTopology (C : Type u) [Category.{v} C] : GrothendieckTopology C :=
  finestTopology (Set.range yoneda.obj)


/-- `yoneda.obj X` is a sheaf for the canonical topology. -/
theorem isSheaf_yoneda_obj (X : C) : Presieve.IsSheaf (canonicalTopology C) (yoneda.obj X) :=
  fun _ _ hS => sheaf_for_finestTopology _ (Set.mem_range_self _) _ hS


/-- A representable functor is a sheaf for the canonical topology. -/
theorem isSheaf_of_isRepresentable (P : Cᵒᵖ ⥤ Type v) [P.IsRepresentable] :
    Presieve.IsSheaf (canonicalTopology C) P :=
  Presieve.isSheaf_iso (canonicalTopology C) P.reprW (isSheaf_yoneda_obj _)


/-- A subcanonical topology is a topology which is smaller than the canonical topology.
Equivalently, a topology is subcanonical iff every representable is a sheaf.
-/
class Subcanonical (J : GrothendieckTopology C) : Prop where
  le_canonical : J ≤ canonicalTopology C


lemma le_canonical (J : GrothendieckTopology C) [Subcanonical J] : J ≤ canonicalTopology C :=
  Subcanonical.le_canonical


instance : (canonicalTopology C).Subcanonical where
  le_canonical := le_rfl


/-- If every functor `yoneda.obj X` is a `J`-sheaf, then `J` is subcanonical. -/
theorem of_isSheaf_yoneda_obj (J : GrothendieckTopology C)
    (h : ∀ X, Presieve.IsSheaf J (yoneda.obj X)) : Subcanonical J where
                                            /-
                                              C : Type u
                                              inst✝ : CategoryTheory.Category.{v, u} C
                                              J : CategoryTheory.GrothendieckTopology C
                                              h : ∀ (X : C), CategoryTheory.Presieve.IsSheaf J (CategoryTheory.yoneda.obj X)
                                              ⊢ ∀ (P : CategoryTheory.Functor (Opposite C) (Type v)), Membership.mem (Set.ra …
                                            -/
  le_canonical := le_finestTopology _ _ (by rintro P ⟨X, rfl⟩; apply h)
                                                               /-
                                                                 🎉 no goals
                                                               -/


/-- If `J` is subcanonical, then any representable is a `J`-sheaf. -/
theorem isSheaf_of_isRepresentable {J : GrothendieckTopology C} [Subcanonical J]
    (P : Cᵒᵖ ⥤ Type v) [P.IsRepresentable] : Presieve.IsSheaf J P :=
  Presieve.isSheaf_of_le _ J.le_canonical (Sheaf.isSheaf_of_isRepresentable P)


/--
If `J` is subcanonical, we obtain a "Yoneda" functor from the defining site
into the sheaf category.
-/
@[simps]
def yoneda [J.Subcanonical] : C ⥤ Sheaf J (Type v) where
  obj X := ⟨CategoryTheory.yoneda.obj X, by
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      inst✝ : J.Subcanonical
      X : C
      ⊢ CategoryTheory.Presheaf.IsSheaf J (CategoryTheory.yoneda.obj X)
    -/
    rw [isSheaf_iff_isSheaf_of_type]
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      inst✝ : J.Subcanonical
      X : C
      ⊢ CategoryTheory.Presieve.IsSheaf J (CategoryTheory.yoneda.obj X)
    -/
    apply Subcanonical.isSheaf_of_isRepresentable⟩
    /-
      🎉 no goals
    -/
  map f := ⟨CategoryTheory.yoneda.map f⟩


/--
The yoneda embedding into the presheaf category factors through the one
to the sheaf category.
-/
def yonedaCompSheafToPresheaf :
    J.yoneda ⋙ sheafToPresheaf J (Type v) ≅ CategoryTheory.yoneda :=
  Iso.refl _


/-- The yoneda functor into the sheaf category is fully faithful -/
def yonedaFullyFaithful : (J.yoneda).FullyFaithful :=
  Functor.FullyFaithful.ofCompFaithful (G := sheafToPresheaf J (Type v)) Yoneda.fullyFaithful


instance : (J.yoneda).Full := (J.yonedaFullyFaithful).full


instance : (J.yoneda).Faithful := (J.yonedaFullyFaithful).faithful


@[deprecated (since := "2024-10-29")] alias Sheaf.Subcanonical := GrothendieckTopology.Subcanonical

@[deprecated (since := "2024-10-29")] alias Sheaf.Subcanonical.of_isSheaf_yoneda_obj :=
  GrothendieckTopology.Subcanonical.of_isSheaf_yoneda_obj

@[deprecated (since := "2024-10-29")] alias Sheaf.Subcanonical.isSheaf_of_isRepresentable :=
  GrothendieckTopology.Subcanonical.isSheaf_of_isRepresentable

@[deprecated (since := "2024-10-29")] alias Sheaf.Subcanonical.yoneda :=
  GrothendieckTopology.yoneda

@[deprecated (since := "2024-10-29")] alias Sheaf.Subcanonical.yonedaCompSheafToPresheaf :=
  GrothendieckTopology.yonedaCompSheafToPresheaf

@[deprecated (since := "2024-10-29")] alias Sheaf.Subcanonical.yonedaFullyFaithful :=
  GrothendieckTopology.yonedaFullyFaithful


