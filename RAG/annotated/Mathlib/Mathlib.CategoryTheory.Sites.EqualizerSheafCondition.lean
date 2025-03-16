/--
The middle object of the fork diagram given in Equation (3) of [MM92], as well as the fork diagram
of <https://stacks.math.columbia.edu/tag/00VM>.
-/
def FirstObj : Type max v u :=
  ∏ᶜ fun f : ΣY, { f : Y ⟶ X // R f } => P.obj (op f.1)


@[ext]
lemma FirstObj.ext (z₁ z₂ : FirstObj P R) (h : ∀ (Y : C) (f : Y ⟶ X)
    (hf : R f), (Pi.π _ ⟨Y, f, hf⟩ : FirstObj P R ⟶ _) z₁ =
      (Pi.π _ ⟨Y, f, hf⟩ : FirstObj P R ⟶ _) z₂) : z₁ = z₂ := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    P : CategoryTheory.Functor (Opposite C) (Type (max v u))
    X : C
    R : CategoryTheory.Presieve X
    z₁ z₂ : CategoryTheory.Equalizer.FirstObj P R
    h : ∀ (Y : C) (f : Quiver.Hom Y X) (hf : R f), Eq (CategoryTheory.Limits.Pi.π  …
    ⊢ Eq z₁ z₂
  -/
  apply Limits.Types.limit_ext
  /-
    case w
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    P : CategoryTheory.Functor (Opposite C) (Type (max v u))
    X : C
    R : CategoryTheory.Presieve X
    z₁ z₂ : CategoryTheory.Equalizer.FirstObj P R
    h : ∀ (Y : C) (f : Quiver.Hom Y X) (hf : R f), Eq (CategoryTheory.Limits.Pi.π  …
    ⊢ ∀ (j : CategoryTheory.Discrete (Sigma fun Y => Subtype fun f => R f)), Eq (C …
  -/
  rintro ⟨⟨Y, f, hf⟩⟩
  /-
    case w.mk.mk.mk
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    P : CategoryTheory.Functor (Opposite C) (Type (max v u))
    X : C
    R : CategoryTheory.Presieve X
    z₁ z₂ : CategoryTheory.Equalizer.FirstObj P R
    h : ∀ (Y : C) (f : Quiver.Hom Y X) (hf : R f), Eq (CategoryTheory.Limits.Pi.π  …
    Y : C
    f : Quiver.Hom Y X
    hf : R f
    ⊢ Eq (CategoryTheory.Limits.limit.π (CategoryTheory.Discrete.functor fun f =>  …
  -/
  exact h Y f hf
  /-
    🎉 no goals
  -/


/-- Show that `FirstObj` is isomorphic to `FamilyOfElements`. -/
@[simps]
def firstObjEqFamily : FirstObj P R ≅ R.FamilyOfElements P where
  hom t _ _ hf := Pi.π (fun f : ΣY, { f : Y ⟶ X // R f } => P.obj (op f.1)) ⟨_, _, hf⟩ t
  inv := Pi.lift fun f x => x _ f.2.2


instance : Inhabited (FirstObj P (⊥ : Presieve X)) :=
  (firstObjEqFamily P _).toEquiv.inhabited

-- Porting note: was not needed in mathlib

instance : Inhabited (FirstObj P ((⊥ : Sieve X) : Presieve X)) :=
  (inferInstance : Inhabited (FirstObj P (⊥ : Presieve X)))


/--
The left morphism of the fork diagram given in Equation (3) of [MM92], as well as the fork diagram
of <https://stacks.math.columbia.edu/tag/00VM>.
-/
def forkMap : P.obj (op X) ⟶ FirstObj P R :=
  Pi.lift fun f => P.map f.2.1.op


/-- The rightmost object of the fork diagram of Equation (3) [MM92], which contains the data used
to check a family is compatible.
-/
def SecondObj : Type max v u :=
  ∏ᶜ fun f : Σ(Y Z : _) (_ : Z ⟶ Y), { f' : Y ⟶ X // S f' } => P.obj (op f.2.1)


@[ext]
lemma SecondObj.ext (z₁ z₂ : SecondObj P S) (h : ∀ (Y Z : C) (g : Z ⟶ Y) (f : Y ⟶ X)
    (hf : S.arrows f), (Pi.π _ ⟨Y, Z, g, f, hf⟩ : SecondObj P S ⟶ _) z₁ =
      (Pi.π _ ⟨Y, Z, g, f, hf⟩ : SecondObj P S ⟶ _) z₂) : z₁ = z₂ := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    P : CategoryTheory.Functor (Opposite C) (Type (max v u))
    X : C
    S : CategoryTheory.Sieve X
    z₁ z₂ : CategoryTheory.Equalizer.Sieve.SecondObj P S
    h : ∀ (Y Z : C) (g : Quiver.Hom Z Y) (f : Quiver.Hom Y X) (hf : S.arrows f), E …
    ⊢ Eq z₁ z₂
  -/
  apply Limits.Types.limit_ext
  /-
    case w
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    P : CategoryTheory.Functor (Opposite C) (Type (max v u))
    X : C
    S : CategoryTheory.Sieve X
    z₁ z₂ : CategoryTheory.Equalizer.Sieve.SecondObj P S
    h : ∀ (Y Z : C) (g : Quiver.Hom Z Y) (f : Quiver.Hom Y X) (hf : S.arrows f), E …
    ⊢ ∀ (j : CategoryTheory.Discrete (Sigma fun Y => Sigma fun Z => Sigma fun x => …
  -/
  rintro ⟨⟨Y, Z, g, f, hf⟩⟩
  /-
    case w.mk.mk.mk.mk.mk
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    P : CategoryTheory.Functor (Opposite C) (Type (max v u))
    X : C
    S : CategoryTheory.Sieve X
    z₁ z₂ : CategoryTheory.Equalizer.Sieve.SecondObj P S
    h : ∀ (Y Z : C) (g : Quiver.Hom Z Y) (f : Quiver.Hom Y X) (hf : S.arrows f), E …
    Y Z : C
    g : Quiver.Hom Z Y
    f : Quiver.Hom Y X
    hf : S.arrows f
    ⊢ Eq (CategoryTheory.Limits.limit.π (CategoryTheory.Discrete.functor fun f =>  …
  -/
  apply h
  /-
    🎉 no goals
  -/


/-- The map `p` of Equations (3,4) [MM92]. -/
def firstMap : FirstObj P (S : Presieve X) ⟶ SecondObj P S :=
  Pi.lift fun fg =>
    Pi.π _ (⟨_, _, S.downward_closed fg.2.2.2.2 fg.2.2.1⟩ : ΣY, { f : Y ⟶ X // S f })


instance : Inhabited (SecondObj P (⊥ : Sieve X)) :=
  ⟨firstMap _ _ default⟩


/-- The map `a` of Equations (3,4) [MM92]. -/
def secondMap : FirstObj P (S : Presieve X) ⟶ SecondObj P S :=
  Pi.lift fun fg => Pi.π _ ⟨_, fg.2.2.2⟩ ≫ P.map fg.2.2.1.op


theorem w : forkMap P (S : Presieve X) ≫ firstMap P S = forkMap P S ≫ secondMap P S := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    P : CategoryTheory.Functor (Opposite C) (Type (max v u))
    X : C
    S : CategoryTheory.Sieve X
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Equalizer.forkMap P S …
  -/
  ext
  /-
    case h.h
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    P : CategoryTheory.Functor (Opposite C) (Type (max v u))
    X : C
    S : CategoryTheory.Sieve X
    a✝ : P.obj { unop := X }
    Y✝ Z✝ : C
    g✝ : Quiver.Hom Z✝ Y✝
    f✝ : Quiver.Hom Y✝ X
    hf✝ : S.arrows f✝
    ⊢ Eq (CategoryTheory.Limits.Pi.π (fun f => P.obj { unop := f.snd.fst }) ⟨Y✝, ⟨ …
  -/
  simp [firstMap, secondMap, forkMap]
  /-
    🎉 no goals
  -/


/--
The family of elements given by `x : FirstObj P S` is compatible iff `firstMap` and `secondMap`
map it to the same point.
-/
theorem compatible_iff (x : FirstObj P S.arrows) :
    ((firstObjEqFamily P S.arrows).hom x).Compatible ↔ firstMap P S x = secondMap P S x := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    P : CategoryTheory.Functor (Opposite C) (Type (max v u))
    X : C
    S : CategoryTheory.Sieve X
    x : CategoryTheory.Equalizer.FirstObj P S.arrows
    ⊢ Iff ((CategoryTheory.Equalizer.firstObjEqFamily P S.arrows).hom x).Compatibl …
  -/
  rw [Presieve.compatible_iff_sieveCompatible]
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    P : CategoryTheory.Functor (Opposite C) (Type (max v u))
    X : C
    S : CategoryTheory.Sieve X
    x : CategoryTheory.Equalizer.FirstObj P S.arrows
    ⊢ Iff ((CategoryTheory.Equalizer.firstObjEqFamily P S.arrows).hom x).SieveComp …
  -/
  constructor
    /-
      case mp
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      P : CategoryTheory.Functor (Opposite C) (Type (max v u))
      X : C
      S : CategoryTheory.Sieve X
      x : CategoryTheory.Equalizer.FirstObj P S.arrows
      ⊢ ((CategoryTheory.Equalizer.firstObjEqFamily P S.arrows).hom x).SieveCompatib …
    -/
  · intro t
    /-
      case mp
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      P : CategoryTheory.Functor (Opposite C) (Type (max v u))
      X : C
      S : CategoryTheory.Sieve X
      x : CategoryTheory.Equalizer.FirstObj P S.arrows
      t : ((CategoryTheory.Equalizer.firstObjEqFamily P S.arrows).hom x).SieveCompat …
      ⊢ Eq (CategoryTheory.Equalizer.Sieve.firstMap P S x) (CategoryTheory.Equalizer …
    -/
    apply SecondObj.ext
    /-
      case mp.h
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      P : CategoryTheory.Functor (Opposite C) (Type (max v u))
      X : C
      S : CategoryTheory.Sieve X
      x : CategoryTheory.Equalizer.FirstObj P S.arrows
      t : ((CategoryTheory.Equalizer.firstObjEqFamily P S.arrows).hom x).SieveCompat …
      ⊢ ∀ (Y Z : C) (g : Quiver.Hom Z Y) (f : Quiver.Hom Y X) (hf : S.arrows f), Eq  …
    -/
    intros Y Z g f hf
    /-
      case mp.h
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      P : CategoryTheory.Functor (Opposite C) (Type (max v u))
      X : C
      S : CategoryTheory.Sieve X
      x : CategoryTheory.Equalizer.FirstObj P S.arrows
      t : ((CategoryTheory.Equalizer.firstObjEqFamily P S.arrows).hom x).SieveCompat …
      Y Z : C
      g : Quiver.Hom Z Y
      f : Quiver.Hom Y X
      hf : S.arrows f
      ⊢ Eq (CategoryTheory.Limits.Pi.π (fun f => P.obj { unop := f.snd.fst }) ⟨Y, ⟨Z …
    -/
    simpa [firstMap, secondMap] using t _ g hf
    /-
      🎉 no goals
    -/
    /-
      case mpr
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      P : CategoryTheory.Functor (Opposite C) (Type (max v u))
      X : C
      S : CategoryTheory.Sieve X
      x : CategoryTheory.Equalizer.FirstObj P S.arrows
      ⊢ Eq (CategoryTheory.Equalizer.Sieve.firstMap P S x) (CategoryTheory.Equalizer …
    -/
  · intro t Y Z f g hf
    /-
      case mpr
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      P : CategoryTheory.Functor (Opposite C) (Type (max v u))
      X : C
      S : CategoryTheory.Sieve X
      x : CategoryTheory.Equalizer.FirstObj P S.arrows
      t : Eq (CategoryTheory.Equalizer.Sieve.firstMap P S x) (CategoryTheory.Equaliz …
      Y Z : C
      f : Quiver.Hom Y X
      g : Quiver.Hom Z Y
      hf : S.arrows f
      ⊢ Eq ((CategoryTheory.Equalizer.firstObjEqFamily P S.arrows).hom x (CategoryTh …
    -/
    rw [Types.limit_ext_iff'] at t
    /-
      case mpr
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      P : CategoryTheory.Functor (Opposite C) (Type (max v u))
      X : C
      S : CategoryTheory.Sieve X
      x : CategoryTheory.Equalizer.FirstObj P S.arrows
      t : ∀ (j : CategoryTheory.Discrete (Sigma fun Y => Sigma fun Z => Sigma fun x  …
      Y Z : C
      f : Quiver.Hom Y X
      g : Quiver.Hom Z Y
      hf : S.arrows f
      ⊢ Eq ((CategoryTheory.Equalizer.firstObjEqFamily P S.arrows).hom x (CategoryTh …
    -/
    simpa [firstMap, secondMap] using t ⟨⟨Y, Z, g, f, hf⟩⟩
    /-
      🎉 no goals
    -/


/-- `P` is a sheaf for `S`, iff the fork given by `w` is an equalizer. -/
theorem equalizer_sheaf_condition :
    Presieve.IsSheafFor P (S : Presieve X) ↔ Nonempty (IsLimit (Fork.ofι _ (w P S))) := by
  rw [Types.type_equalizer_iff_unique,
    ← Equiv.forall_congr_right (firstObjEqFamily P (S : Presieve X)).toEquiv.symm]
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    P : CategoryTheory.Functor (Opposite C) (Type (max v u))
    X : C
    S : CategoryTheory.Sieve X
    ⊢ Iff (CategoryTheory.Presieve.IsSheafFor P S.arrows) (∀ (a : CategoryTheory.P …
  -/
  simp_rw [← compatible_iff]
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    P : CategoryTheory.Functor (Opposite C) (Type (max v u))
    X : C
    S : CategoryTheory.Sieve X
    ⊢ Iff (CategoryTheory.Presieve.IsSheafFor P S.arrows) (∀ (a : CategoryTheory.P …
  -/
  simp only [inv_hom_id_apply, Iso.toEquiv_symm_fun]
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    P : CategoryTheory.Functor (Opposite C) (Type (max v u))
    X : C
    S : CategoryTheory.Sieve X
    ⊢ Iff (CategoryTheory.Presieve.IsSheafFor P S.arrows) (∀ (a : CategoryTheory.P …
  -/
  apply forall₂_congr
  /-
    case h
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    P : CategoryTheory.Functor (Opposite C) (Type (max v u))
    X : C
    S : CategoryTheory.Sieve X
    ⊢ ∀ (a : CategoryTheory.Presieve.FamilyOfElements P S.arrows), a.Compatible →  …
  -/
  intro x _
  /-
    case h
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    P : CategoryTheory.Functor (Opposite C) (Type (max v u))
    X : C
    S : CategoryTheory.Sieve X
    x : CategoryTheory.Presieve.FamilyOfElements P S.arrows
    b✝ : x.Compatible
    ⊢ Iff (ExistsUnique fun t => x.IsAmalgamation t) (ExistsUnique fun x_1 => Eq ( …
  -/
  apply existsUnique_congr
  /-
    case h.h
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    P : CategoryTheory.Functor (Opposite C) (Type (max v u))
    X : C
    S : CategoryTheory.Sieve X
    x : CategoryTheory.Presieve.FamilyOfElements P S.arrows
    b✝ : x.Compatible
    ⊢ ∀ (a : P.obj { unop := X }), Iff (x.IsAmalgamation a) (Eq (CategoryTheory.Eq …
  -/
  intro t
  /-
    case h.h
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    P : CategoryTheory.Functor (Opposite C) (Type (max v u))
    X : C
    S : CategoryTheory.Sieve X
    x : CategoryTheory.Presieve.FamilyOfElements P S.arrows
    b✝ : x.Compatible
    t : P.obj { unop := X }
    ⊢ Iff (x.IsAmalgamation t) (Eq (CategoryTheory.Equalizer.forkMap P S.arrows t) …
  -/
  rw [← Iso.toEquiv_symm_fun]
  /-
    case h.h
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    P : CategoryTheory.Functor (Opposite C) (Type (max v u))
    X : C
    S : CategoryTheory.Sieve X
    x : CategoryTheory.Presieve.FamilyOfElements P S.arrows
    b✝ : x.Compatible
    t : P.obj { unop := X }
    ⊢ Iff (x.IsAmalgamation t) (Eq (CategoryTheory.Equalizer.forkMap P S.arrows t) …
  -/
  rw [Equiv.eq_symm_apply]
  /-
    case h.h
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    P : CategoryTheory.Functor (Opposite C) (Type (max v u))
    X : C
    S : CategoryTheory.Sieve X
    x : CategoryTheory.Presieve.FamilyOfElements P S.arrows
    b✝ : x.Compatible
    t : P.obj { unop := X }
    ⊢ Iff (x.IsAmalgamation t) (Eq ((CategoryTheory.Equalizer.firstObjEqFamily P S …
  -/
  constructor
    /-
      case h.h.mp
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      P : CategoryTheory.Functor (Opposite C) (Type (max v u))
      X : C
      S : CategoryTheory.Sieve X
      x : CategoryTheory.Presieve.FamilyOfElements P S.arrows
      b✝ : x.Compatible
      t : P.obj { unop := X }
      ⊢ x.IsAmalgamation t → Eq ((CategoryTheory.Equalizer.firstObjEqFamily P S.arro …
    -/
  · intro q
    /-
      case h.h.mp
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      P : CategoryTheory.Functor (Opposite C) (Type (max v u))
      X : C
      S : CategoryTheory.Sieve X
      x : CategoryTheory.Presieve.FamilyOfElements P S.arrows
      b✝ : x.Compatible
      t : P.obj { unop := X }
      q : x.IsAmalgamation t
      ⊢ Eq ((CategoryTheory.Equalizer.firstObjEqFamily P S.arrows).toEquiv (Category …
    -/
    funext Y f hf
    /-
      case h.h.mp.h.h.h
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      P : CategoryTheory.Functor (Opposite C) (Type (max v u))
      X : C
      S : CategoryTheory.Sieve X
      x : CategoryTheory.Presieve.FamilyOfElements P S.arrows
      b✝ : x.Compatible
      t : P.obj { unop := X }
      q : x.IsAmalgamation t
      Y : C
      f : Quiver.Hom Y X
      hf : S.arrows f
      ⊢ Eq ((CategoryTheory.Equalizer.firstObjEqFamily P S.arrows).toEquiv (Category …
    -/
    simpa [firstObjEqFamily, forkMap] using q _ _
    /-
      🎉 no goals
    -/
    /-
      case h.h.mpr
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      P : CategoryTheory.Functor (Opposite C) (Type (max v u))
      X : C
      S : CategoryTheory.Sieve X
      x : CategoryTheory.Presieve.FamilyOfElements P S.arrows
      b✝ : x.Compatible
      t : P.obj { unop := X }
      ⊢ Eq ((CategoryTheory.Equalizer.firstObjEqFamily P S.arrows).toEquiv (Category …
    -/
  · intro q Y f hf
    /-
      case h.h.mpr
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      P : CategoryTheory.Functor (Opposite C) (Type (max v u))
      X : C
      S : CategoryTheory.Sieve X
      x : CategoryTheory.Presieve.FamilyOfElements P S.arrows
      b✝ : x.Compatible
      t : P.obj { unop := X }
      q : Eq ((CategoryTheory.Equalizer.firstObjEqFamily P S.arrows).toEquiv (Catego …
      Y : C
      f : Quiver.Hom Y X
      hf : S.arrows f
      ⊢ Eq (P.map f.op t) (x f hf)
    -/
    rw [← q]
    /-
      case h.h.mpr
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      P : CategoryTheory.Functor (Opposite C) (Type (max v u))
      X : C
      S : CategoryTheory.Sieve X
      x : CategoryTheory.Presieve.FamilyOfElements P S.arrows
      b✝ : x.Compatible
      t : P.obj { unop := X }
      q : Eq ((CategoryTheory.Equalizer.firstObjEqFamily P S.arrows).toEquiv (Catego …
      Y : C
      f : Quiver.Hom Y X
      hf : S.arrows f
      ⊢ Eq (P.map f.op t) ((CategoryTheory.Equalizer.firstObjEqFamily P S.arrows).to …
    -/
    simp [firstObjEqFamily, forkMap]
    /-
      🎉 no goals
    -/


/--
The rightmost object of the fork diagram of https://stacks.math.columbia.edu/tag/00VM, which
contains the data used to check a family of elements for a presieve is compatible.
-/
@[simp] def SecondObj : Type max v u :=
  ∏ᶜ fun fg : (ΣY, { f : Y ⟶ X // R f }) × ΣZ, { g : Z ⟶ X // R g } =>
    haveI := Presieve.hasPullbacks.has_pullbacks fg.1.2.2 fg.2.2.2
    P.obj (op (pullback fg.1.2.1 fg.2.2.1))


/-- The map `pr₀*` of <https://stacks.math.columbia.edu/tag/00VL>. -/
def firstMap : FirstObj P R ⟶ SecondObj P R :=
  Pi.lift fun fg =>
    haveI := Presieve.hasPullbacks.has_pullbacks fg.1.2.2 fg.2.2.2
    Pi.π _ _ ≫ P.map (pullback.fst _ _).op


instance [HasPullbacks C] : Inhabited (SecondObj P (⊥ : Presieve X)) :=
  ⟨firstMap _ _ default⟩


/-- The map `pr₁*` of <https://stacks.math.columbia.edu/tag/00VL>. -/
def secondMap : FirstObj P R ⟶ SecondObj P R :=
  Pi.lift fun fg =>
    haveI := Presieve.hasPullbacks.has_pullbacks fg.1.2.2 fg.2.2.2
    Pi.π _ _ ≫ P.map (pullback.snd _ _).op


theorem w : forkMap P R ≫ firstMap P R = forkMap P R ≫ secondMap P R := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    P : CategoryTheory.Functor (Opposite C) (Type (max v u))
    X : C
    R : CategoryTheory.Presieve X
    inst✝ : R.hasPullbacks
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Equalizer.forkMap P R …
  -/
  dsimp
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    P : CategoryTheory.Functor (Opposite C) (Type (max v u))
    X : C
    R : CategoryTheory.Presieve X
    inst✝ : R.hasPullbacks
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Equalizer.forkMap P R …
  -/
  ext fg
  /-
    case h.h
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    P : CategoryTheory.Functor (Opposite C) (Type (max v u))
    X : C
    R : CategoryTheory.Presieve X
    inst✝ : R.hasPullbacks
    fg : Prod (Sigma fun Y => Subtype fun f => R f) (Sigma fun Z => Subtype fun g  …
    a✝ : P.obj { unop := X }
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  simp only [firstMap, secondMap, forkMap]
  /-
    case h.h
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    P : CategoryTheory.Functor (Opposite C) (Type (max v u))
    X : C
    R : CategoryTheory.Presieve X
    inst✝ : R.hasPullbacks
    fg : Prod (Sigma fun Y => Subtype fun f => R f) (Sigma fun Z => Subtype fun g  …
    a✝ : P.obj { unop := X }
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  simp only [limit.lift_π, limit.lift_π_assoc, assoc, Fan.mk_π_app]
  /-
    case h.h
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    P : CategoryTheory.Functor (Opposite C) (Type (max v u))
    X : C
    R : CategoryTheory.Presieve X
    inst✝ : R.hasPullbacks
    fg : Prod (Sigma fun Y => Subtype fun f => R f) (Sigma fun Z => Subtype fun g  …
    a✝ : P.obj { unop := X }
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (P.map (↑fg.1.snd).op) (P.map (Catego …
  -/
  haveI := Presieve.hasPullbacks.has_pullbacks fg.1.2.2 fg.2.2.2
  /-
    case h.h
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    P : CategoryTheory.Functor (Opposite C) (Type (max v u))
    X : C
    R : CategoryTheory.Presieve X
    inst✝ : R.hasPullbacks
    fg : Prod (Sigma fun Y => Subtype fun f => R f) (Sigma fun Z => Subtype fun g  …
    a✝ : P.obj { unop := X }
    this : CategoryTheory.Limits.HasPullback ↑fg.1.snd ↑fg.2.snd
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (P.map (↑fg.1.snd).op) (P.map (Catego …
  -/
  rw [← P.map_comp, ← op_comp, pullback.condition]
  /-
    case h.h
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    P : CategoryTheory.Functor (Opposite C) (Type (max v u))
    X : C
    R : CategoryTheory.Presieve X
    inst✝ : R.hasPullbacks
    fg : Prod (Sigma fun Y => Subtype fun f => R f) (Sigma fun Z => Subtype fun g  …
    a✝ : P.obj { unop := X }
    this : CategoryTheory.Limits.HasPullback ↑fg.1.snd ↑fg.2.snd
    ⊢ Eq (P.map (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullbac …
  -/
  simp
  /-
    🎉 no goals
  -/


/--
The family of elements given by `x : FirstObj P S` is compatible iff `firstMap` and `secondMap`
map it to the same point.
-/
theorem compatible_iff (x : FirstObj P R) :
    ((firstObjEqFamily P R).hom x).Compatible ↔ firstMap P R x = secondMap P R x := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    P : CategoryTheory.Functor (Opposite C) (Type (max v u))
    X : C
    R : CategoryTheory.Presieve X
    inst✝ : R.hasPullbacks
    x : CategoryTheory.Equalizer.FirstObj P R
    ⊢ Iff ((CategoryTheory.Equalizer.firstObjEqFamily P R).hom x).Compatible (Eq ( …
  -/
  rw [Presieve.pullbackCompatible_iff]
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    P : CategoryTheory.Functor (Opposite C) (Type (max v u))
    X : C
    R : CategoryTheory.Presieve X
    inst✝ : R.hasPullbacks
    x : CategoryTheory.Equalizer.FirstObj P R
    ⊢ Iff ((CategoryTheory.Equalizer.firstObjEqFamily P R).hom x).PullbackCompatib …
  -/
  constructor
    /-
      case mp
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      P : CategoryTheory.Functor (Opposite C) (Type (max v u))
      X : C
      R : CategoryTheory.Presieve X
      inst✝ : R.hasPullbacks
      x : CategoryTheory.Equalizer.FirstObj P R
      ⊢ ((CategoryTheory.Equalizer.firstObjEqFamily P R).hom x).PullbackCompatible → …
    -/
  · intro t
    /-
      case mp
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      P : CategoryTheory.Functor (Opposite C) (Type (max v u))
      X : C
      R : CategoryTheory.Presieve X
      inst✝ : R.hasPullbacks
      x : CategoryTheory.Equalizer.FirstObj P R
      t : ((CategoryTheory.Equalizer.firstObjEqFamily P R).hom x).PullbackCompatible
      ⊢ Eq (CategoryTheory.Equalizer.Presieve.firstMap P R x) (CategoryTheory.Equali …
    -/
    apply Limits.Types.limit_ext
    /-
      case mp.w
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      P : CategoryTheory.Functor (Opposite C) (Type (max v u))
      X : C
      R : CategoryTheory.Presieve X
      inst✝ : R.hasPullbacks
      x : CategoryTheory.Equalizer.FirstObj P R
      t : ((CategoryTheory.Equalizer.firstObjEqFamily P R).hom x).PullbackCompatible
      ⊢ ∀ (j : CategoryTheory.Discrete (Prod (Sigma fun Y => Subtype fun f => R f) ( …
    -/
    rintro ⟨⟨Y, f, hf⟩, Z, g, hg⟩
    /-
      case mp.w.mk.mk.mk.mk.mk.mk
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      P : CategoryTheory.Functor (Opposite C) (Type (max v u))
      X : C
      R : CategoryTheory.Presieve X
      inst✝ : R.hasPullbacks
      x : CategoryTheory.Equalizer.FirstObj P R
      t : ((CategoryTheory.Equalizer.firstObjEqFamily P R).hom x).PullbackCompatible
      Y : C
      f : Quiver.Hom Y X
      hf : R f
      Z : C
      g : Quiver.Hom Z X
      hg : R g
      ⊢ Eq (CategoryTheory.Limits.limit.π (CategoryTheory.Discrete.functor fun fg => …
    -/
    simpa [firstMap, secondMap] using t hf hg
    /-
      🎉 no goals
    -/
    /-
      case mpr
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      P : CategoryTheory.Functor (Opposite C) (Type (max v u))
      X : C
      R : CategoryTheory.Presieve X
      inst✝ : R.hasPullbacks
      x : CategoryTheory.Equalizer.FirstObj P R
      ⊢ Eq (CategoryTheory.Equalizer.Presieve.firstMap P R x) (CategoryTheory.Equali …
    -/
  · intro t Y Z f g hf hg
    /-
      case mpr
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      P : CategoryTheory.Functor (Opposite C) (Type (max v u))
      X : C
      R : CategoryTheory.Presieve X
      inst✝ : R.hasPullbacks
      x : CategoryTheory.Equalizer.FirstObj P R
      t : Eq (CategoryTheory.Equalizer.Presieve.firstMap P R x) (CategoryTheory.Equa …
      Y Z : C
      f : Quiver.Hom Y X
      g : Quiver.Hom Z X
      hf : R f
      hg : R g
      ⊢ Eq (P.map (CategoryTheory.Limits.pullback.fst f g).op ((CategoryTheory.Equal …
    -/
    rw [Types.limit_ext_iff'] at t
    /-
      case mpr
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      P : CategoryTheory.Functor (Opposite C) (Type (max v u))
      X : C
      R : CategoryTheory.Presieve X
      inst✝ : R.hasPullbacks
      x : CategoryTheory.Equalizer.FirstObj P R
      t : ∀ (j : CategoryTheory.Discrete (Prod (Sigma fun Y => Subtype fun f => R f) …
      Y Z : C
      f : Quiver.Hom Y X
      g : Quiver.Hom Z X
      hf : R f
      hg : R g
      ⊢ Eq (P.map (CategoryTheory.Limits.pullback.fst f g).op ((CategoryTheory.Equal …
    -/
    simpa [firstMap, secondMap] using t ⟨⟨⟨Y, f, hf⟩, Z, g, hg⟩⟩
    /-
      🎉 no goals
    -/


/-- `P` is a sheaf for `R`, iff the fork given by `w` is an equalizer.
See <https://stacks.math.columbia.edu/tag/00VM>.
-/
theorem sheaf_condition : R.IsSheafFor P ↔ Nonempty (IsLimit (Fork.ofι _ (w P R))) := by
  rw [Types.type_equalizer_iff_unique,
    ← Equiv.forall_congr_right (firstObjEqFamily P R).toEquiv.symm]
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    P : CategoryTheory.Functor (Opposite C) (Type (max v u))
    X : C
    R : CategoryTheory.Presieve X
    inst✝ : R.hasPullbacks
    ⊢ Iff (CategoryTheory.Presieve.IsSheafFor P R) (∀ (a : CategoryTheory.Presieve …
  -/
  simp_rw [← compatible_iff, ← Iso.toEquiv_fun, Equiv.apply_symm_apply]
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    P : CategoryTheory.Functor (Opposite C) (Type (max v u))
    X : C
    R : CategoryTheory.Presieve X
    inst✝ : R.hasPullbacks
    ⊢ Iff (CategoryTheory.Presieve.IsSheafFor P R) (∀ (a : CategoryTheory.Presieve …
  -/
  apply forall₂_congr
  /-
    case h
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    P : CategoryTheory.Functor (Opposite C) (Type (max v u))
    X : C
    R : CategoryTheory.Presieve X
    inst✝ : R.hasPullbacks
    ⊢ ∀ (a : CategoryTheory.Presieve.FamilyOfElements P R), a.Compatible → Iff (Ex …
  -/
  intro x _
  /-
    case h
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    P : CategoryTheory.Functor (Opposite C) (Type (max v u))
    X : C
    R : CategoryTheory.Presieve X
    inst✝ : R.hasPullbacks
    x : CategoryTheory.Presieve.FamilyOfElements P R
    b✝ : x.Compatible
    ⊢ Iff (ExistsUnique fun t => x.IsAmalgamation t) (ExistsUnique fun x_1 => Eq ( …
  -/
  apply existsUnique_congr
  /-
    case h.h
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    P : CategoryTheory.Functor (Opposite C) (Type (max v u))
    X : C
    R : CategoryTheory.Presieve X
    inst✝ : R.hasPullbacks
    x : CategoryTheory.Presieve.FamilyOfElements P R
    b✝ : x.Compatible
    ⊢ ∀ (a : P.obj { unop := X }), Iff (x.IsAmalgamation a) (Eq (CategoryTheory.Eq …
  -/
  intro t
  /-
    case h.h
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    P : CategoryTheory.Functor (Opposite C) (Type (max v u))
    X : C
    R : CategoryTheory.Presieve X
    inst✝ : R.hasPullbacks
    x : CategoryTheory.Presieve.FamilyOfElements P R
    b✝ : x.Compatible
    t : P.obj { unop := X }
    ⊢ Iff (x.IsAmalgamation t) (Eq (CategoryTheory.Equalizer.forkMap P R t) ((Cate …
  -/
  rw [Equiv.eq_symm_apply]
  /-
    case h.h
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    P : CategoryTheory.Functor (Opposite C) (Type (max v u))
    X : C
    R : CategoryTheory.Presieve X
    inst✝ : R.hasPullbacks
    x : CategoryTheory.Presieve.FamilyOfElements P R
    b✝ : x.Compatible
    t : P.obj { unop := X }
    ⊢ Iff (x.IsAmalgamation t) (Eq ((CategoryTheory.Equalizer.firstObjEqFamily P R …
  -/
  constructor
    /-
      case h.h.mp
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      P : CategoryTheory.Functor (Opposite C) (Type (max v u))
      X : C
      R : CategoryTheory.Presieve X
      inst✝ : R.hasPullbacks
      x : CategoryTheory.Presieve.FamilyOfElements P R
      b✝ : x.Compatible
      t : P.obj { unop := X }
      ⊢ x.IsAmalgamation t → Eq ((CategoryTheory.Equalizer.firstObjEqFamily P R).toE …
    -/
  · intro q
    /-
      case h.h.mp
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      P : CategoryTheory.Functor (Opposite C) (Type (max v u))
      X : C
      R : CategoryTheory.Presieve X
      inst✝ : R.hasPullbacks
      x : CategoryTheory.Presieve.FamilyOfElements P R
      b✝ : x.Compatible
      t : P.obj { unop := X }
      q : x.IsAmalgamation t
      ⊢ Eq ((CategoryTheory.Equalizer.firstObjEqFamily P R).toEquiv (CategoryTheory. …
    -/
    funext Y f hf
    /-
      case h.h.mp.h.h.h
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      P : CategoryTheory.Functor (Opposite C) (Type (max v u))
      X : C
      R : CategoryTheory.Presieve X
      inst✝ : R.hasPullbacks
      x : CategoryTheory.Presieve.FamilyOfElements P R
      b✝ : x.Compatible
      t : P.obj { unop := X }
      q : x.IsAmalgamation t
      Y : C
      f : Quiver.Hom Y X
      hf : R f
      ⊢ Eq ((CategoryTheory.Equalizer.firstObjEqFamily P R).toEquiv (CategoryTheory. …
    -/
    simpa [forkMap] using q _ _
    /-
      🎉 no goals
    -/
    /-
      case h.h.mpr
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      P : CategoryTheory.Functor (Opposite C) (Type (max v u))
      X : C
      R : CategoryTheory.Presieve X
      inst✝ : R.hasPullbacks
      x : CategoryTheory.Presieve.FamilyOfElements P R
      b✝ : x.Compatible
      t : P.obj { unop := X }
      ⊢ Eq ((CategoryTheory.Equalizer.firstObjEqFamily P R).toEquiv (CategoryTheory. …
    -/
  · intro q Y f hf
    /-
      case h.h.mpr
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      P : CategoryTheory.Functor (Opposite C) (Type (max v u))
      X : C
      R : CategoryTheory.Presieve X
      inst✝ : R.hasPullbacks
      x : CategoryTheory.Presieve.FamilyOfElements P R
      b✝ : x.Compatible
      t : P.obj { unop := X }
      q : Eq ((CategoryTheory.Equalizer.firstObjEqFamily P R).toEquiv (CategoryTheor …
      Y : C
      f : Quiver.Hom Y X
      hf : R f
      ⊢ Eq (P.map f.op t) (x f hf)
    -/
    rw [← q]
    /-
      case h.h.mpr
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      P : CategoryTheory.Functor (Opposite C) (Type (max v u))
      X : C
      R : CategoryTheory.Presieve X
      inst✝ : R.hasPullbacks
      x : CategoryTheory.Presieve.FamilyOfElements P R
      b✝ : x.Compatible
      t : P.obj { unop := X }
      q : Eq ((CategoryTheory.Equalizer.firstObjEqFamily P R).toEquiv (CategoryTheor …
      Y : C
      f : Quiver.Hom Y X
      hf : R f
      ⊢ Eq (P.map f.op t) ((CategoryTheory.Equalizer.firstObjEqFamily P R).toEquiv ( …
    -/
    simp [forkMap]
    /-
      🎉 no goals
    -/


/--
The middle object of the fork diagram of <https://stacks.math.columbia.edu/tag/00VM>.
The difference between this and `Equalizer.FirstObj P (ofArrows X π)` arises if the family of
arrows `π` contains duplicates. The `Presieve.ofArrows` doesn't see those.
-/
def FirstObj : Type w := ∏ᶜ (fun i ↦ P.obj (op (X i)))


@[ext]
lemma FirstObj.ext (z₁ z₂ : FirstObj P X) (h : ∀ i, (Pi.π _ i : FirstObj P X ⟶ _) z₁ =
    (Pi.π _ i : FirstObj P X ⟶ _) z₂) : z₁ = z₂ := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    P : CategoryTheory.Functor (Opposite C) (Type w)
    I : Type
    X : I → C
    z₁ z₂ : CategoryTheory.Equalizer.Presieve.Arrows.FirstObj P X
    h : ∀ (i : I), Eq (CategoryTheory.Limits.Pi.π (fun i => P.obj { unop := X i }) …
    ⊢ Eq z₁ z₂
  -/
  apply Limits.Types.limit_ext
  /-
    case w
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    P : CategoryTheory.Functor (Opposite C) (Type w)
    I : Type
    X : I → C
    z₁ z₂ : CategoryTheory.Equalizer.Presieve.Arrows.FirstObj P X
    h : ∀ (i : I), Eq (CategoryTheory.Limits.Pi.π (fun i => P.obj { unop := X i }) …
    ⊢ ∀ (j : CategoryTheory.Discrete I), Eq (CategoryTheory.Limits.limit.π (Catego …
  -/
  rintro ⟨i⟩
  /-
    case w.mk
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    P : CategoryTheory.Functor (Opposite C) (Type w)
    I : Type
    X : I → C
    z₁ z₂ : CategoryTheory.Equalizer.Presieve.Arrows.FirstObj P X
    h : ∀ (i : I), Eq (CategoryTheory.Limits.Pi.π (fun i => P.obj { unop := X i }) …
    i : I
    ⊢ Eq (CategoryTheory.Limits.limit.π (CategoryTheory.Discrete.functor fun i =>  …
  -/
  exact h i
  /-
    🎉 no goals
  -/


/--
The rightmost object of the fork diagram of https://stacks.math.columbia.edu/tag/00VM.
The difference between this and `Equalizer.Presieve.SecondObj P (ofArrows X π)` arises if the
family of arrows `π` contains duplicates. The `Presieve.ofArrows` doesn't see those.
-/
def SecondObj : Type w  :=
  ∏ᶜ (fun (ij : I × I) ↦ P.obj (op (pullback (π ij.1) (π ij.2))))


@[ext]
lemma SecondObj.ext (z₁ z₂ : SecondObj P X π) (h : ∀ ij, (Pi.π _ ij : SecondObj P X π ⟶ _) z₁ =
    (Pi.π _ ij : SecondObj P X π ⟶ _) z₂) : z₁ = z₂ := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    P : CategoryTheory.Functor (Opposite C) (Type w)
    B : C
    I : Type
    X : I → C
    π : (i : I) → Quiver.Hom (X i) B
    inst✝ : (CategoryTheory.Presieve.ofArrows X π).hasPullbacks
    z₁ z₂ : CategoryTheory.Equalizer.Presieve.Arrows.SecondObj P X π
    h : ∀ (ij : Prod I I), Eq (CategoryTheory.Limits.Pi.π (fun ij => P.obj { unop  …
    ⊢ Eq z₁ z₂
  -/
  apply Limits.Types.limit_ext
  /-
    case w
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    P : CategoryTheory.Functor (Opposite C) (Type w)
    B : C
    I : Type
    X : I → C
    π : (i : I) → Quiver.Hom (X i) B
    inst✝ : (CategoryTheory.Presieve.ofArrows X π).hasPullbacks
    z₁ z₂ : CategoryTheory.Equalizer.Presieve.Arrows.SecondObj P X π
    h : ∀ (ij : Prod I I), Eq (CategoryTheory.Limits.Pi.π (fun ij => P.obj { unop  …
    ⊢ ∀ (j : CategoryTheory.Discrete (Prod I I)), Eq (CategoryTheory.Limits.limit. …
  -/
  rintro ⟨i⟩
  /-
    case w.mk
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    P : CategoryTheory.Functor (Opposite C) (Type w)
    B : C
    I : Type
    X : I → C
    π : (i : I) → Quiver.Hom (X i) B
    inst✝ : (CategoryTheory.Presieve.ofArrows X π).hasPullbacks
    z₁ z₂ : CategoryTheory.Equalizer.Presieve.Arrows.SecondObj P X π
    h : ∀ (ij : Prod I I), Eq (CategoryTheory.Limits.Pi.π (fun ij => P.obj { unop  …
    i : Prod I I
    ⊢ Eq (CategoryTheory.Limits.limit.π (CategoryTheory.Discrete.functor fun ij => …
  -/
  exact h i
  /-
    🎉 no goals
  -/


/--
The left morphism of the fork diagram.
-/
def forkMap : P.obj (op B) ⟶ FirstObj P X := Pi.lift (fun i ↦ P.map (π i).op)


/--
The first of the two parallel morphisms of the fork diagram, induced by the first projection in
each pullback.
-/
def firstMap : FirstObj P X ⟶ SecondObj P X π :=
  Pi.lift fun _ => Pi.π _ _ ≫ P.map (pullback.fst _ _).op


/--
The second of the two parallel morphisms of the fork diagram, induced by the second projection in
each pullback.
-/
def secondMap : FirstObj P X ⟶ SecondObj P X π :=
  Pi.lift fun _ => Pi.π _ _ ≫ P.map (pullback.snd _ _).op


theorem w : forkMap P X π ≫ firstMap P X π = forkMap P X π ≫ secondMap P X π := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    P : CategoryTheory.Functor (Opposite C) (Type w)
    B : C
    I : Type
    X : I → C
    π : (i : I) → Quiver.Hom (X i) B
    inst✝ : (CategoryTheory.Presieve.ofArrows X π).hasPullbacks
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Equalizer.Presieve.Ar …
  -/
  ext x ij
  simp only [firstMap, secondMap, forkMap, types_comp_apply, Types.pi_lift_π_apply,
    ← FunctorToTypes.map_comp_apply, ← op_comp, pullback.condition]


/--
The family of elements given by `x : FirstObj P S` is compatible iff `firstMap` and `secondMap`
map it to the same point.
-/
theorem compatible_iff (x : FirstObj P X) : (Arrows.Compatible P π ((Types.productIso _).hom x)) ↔
    firstMap P X π x = secondMap P X π x := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    P : CategoryTheory.Functor (Opposite C) (Type w)
    B : C
    I : Type
    X : I → C
    π : (i : I) → Quiver.Hom (X i) B
    inst✝ : (CategoryTheory.Presieve.ofArrows X π).hasPullbacks
    x : CategoryTheory.Equalizer.Presieve.Arrows.FirstObj P X
    ⊢ Iff (CategoryTheory.Presieve.Arrows.Compatible P π ((CategoryTheory.Limits.T …
  -/
  rw [Arrows.pullbackCompatible_iff]
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    P : CategoryTheory.Functor (Opposite C) (Type w)
    B : C
    I : Type
    X : I → C
    π : (i : I) → Quiver.Hom (X i) B
    inst✝ : (CategoryTheory.Presieve.ofArrows X π).hasPullbacks
    x : CategoryTheory.Equalizer.Presieve.Arrows.FirstObj P X
    ⊢ Iff (CategoryTheory.Presieve.Arrows.PullbackCompatible P π ((CategoryTheory. …
  -/
  constructor
    /-
      case mp
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      P : CategoryTheory.Functor (Opposite C) (Type w)
      B : C
      I : Type
      X : I → C
      π : (i : I) → Quiver.Hom (X i) B
      inst✝ : (CategoryTheory.Presieve.ofArrows X π).hasPullbacks
      x : CategoryTheory.Equalizer.Presieve.Arrows.FirstObj P X
      ⊢ CategoryTheory.Presieve.Arrows.PullbackCompatible P π ((CategoryTheory.Limit …
    -/
  · intro t
    /-
      case mp
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      P : CategoryTheory.Functor (Opposite C) (Type w)
      B : C
      I : Type
      X : I → C
      π : (i : I) → Quiver.Hom (X i) B
      inst✝ : (CategoryTheory.Presieve.ofArrows X π).hasPullbacks
      x : CategoryTheory.Equalizer.Presieve.Arrows.FirstObj P X
      t : CategoryTheory.Presieve.Arrows.PullbackCompatible P π ((CategoryTheory.Lim …
      ⊢ Eq (CategoryTheory.Equalizer.Presieve.Arrows.firstMap P X π x) (CategoryTheo …
    -/
    ext ij
    /-
      case mp.h
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      P : CategoryTheory.Functor (Opposite C) (Type w)
      B : C
      I : Type
      X : I → C
      π : (i : I) → Quiver.Hom (X i) B
      inst✝ : (CategoryTheory.Presieve.ofArrows X π).hasPullbacks
      x : CategoryTheory.Equalizer.Presieve.Arrows.FirstObj P X
      t : CategoryTheory.Presieve.Arrows.PullbackCompatible P π ((CategoryTheory.Lim …
      ij : Prod I I
      ⊢ Eq (CategoryTheory.Limits.Pi.π (fun ij => P.obj { unop := CategoryTheory.Lim …
    -/
    simpa [firstMap, secondMap] using t ij.1 ij.2
    /-
      🎉 no goals
    -/
    /-
      case mpr
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      P : CategoryTheory.Functor (Opposite C) (Type w)
      B : C
      I : Type
      X : I → C
      π : (i : I) → Quiver.Hom (X i) B
      inst✝ : (CategoryTheory.Presieve.ofArrows X π).hasPullbacks
      x : CategoryTheory.Equalizer.Presieve.Arrows.FirstObj P X
      ⊢ Eq (CategoryTheory.Equalizer.Presieve.Arrows.firstMap P X π x) (CategoryTheo …
    -/
  · intro t i j
    /-
      case mpr
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      P : CategoryTheory.Functor (Opposite C) (Type w)
      B : C
      I : Type
      X : I → C
      π : (i : I) → Quiver.Hom (X i) B
      inst✝ : (CategoryTheory.Presieve.ofArrows X π).hasPullbacks
      x : CategoryTheory.Equalizer.Presieve.Arrows.FirstObj P X
      t : Eq (CategoryTheory.Equalizer.Presieve.Arrows.firstMap P X π x) (CategoryTh …
      i j : I
      ⊢ Eq (P.map (CategoryTheory.Limits.pullback.fst (π i) (π j)).op ((CategoryTheo …
    -/
    apply_fun Pi.π (fun (ij : I × I) ↦ P.obj (op (pullback (π ij.1) (π ij.2)))) ⟨i, j⟩ at t
    /-
      case mpr
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      P : CategoryTheory.Functor (Opposite C) (Type w)
      B : C
      I : Type
      X : I → C
      π : (i : I) → Quiver.Hom (X i) B
      inst✝ : (CategoryTheory.Presieve.ofArrows X π).hasPullbacks
      x : CategoryTheory.Equalizer.Presieve.Arrows.FirstObj P X
      i j : I
      t : Eq (CategoryTheory.Limits.Pi.π (fun ij => P.obj { unop := CategoryTheory.L …
      ⊢ Eq (P.map (CategoryTheory.Limits.pullback.fst (π i) (π j)).op ((CategoryTheo …
    -/
    simpa [firstMap, secondMap] using t
    /-
      🎉 no goals
    -/


/--
`P` is a sheaf for `Presieve.ofArrows X π`, iff the fork given by `w` is an equalizer.
See <https://stacks.math.columbia.edu/tag/00VM>.
-/
theorem sheaf_condition : (Presieve.ofArrows X π).IsSheafFor P ↔
    Nonempty (IsLimit (Fork.ofι (forkMap P X π) (w P X π))) := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    P : CategoryTheory.Functor (Opposite C) (Type w)
    B : C
    I : Type
    X : I → C
    π : (i : I) → Quiver.Hom (X i) B
    inst✝ : (CategoryTheory.Presieve.ofArrows X π).hasPullbacks
    ⊢ Iff (CategoryTheory.Presieve.IsSheafFor P (CategoryTheory.Presieve.ofArrows  …
  -/
  rw [Types.type_equalizer_iff_unique, isSheafFor_arrows_iff]
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    P : CategoryTheory.Functor (Opposite C) (Type w)
    B : C
    I : Type
    X : I → C
    π : (i : I) → Quiver.Hom (X i) B
    inst✝ : (CategoryTheory.Presieve.ofArrows X π).hasPullbacks
    ⊢ Iff (∀ (x : (i : I) → P.obj { unop := X i }), CategoryTheory.Presieve.Arrows …
  -/
  erw [← Equiv.forall_congr_right (Types.productIso _).toEquiv.symm]
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    P : CategoryTheory.Functor (Opposite C) (Type w)
    B : C
    I : Type
    X : I → C
    π : (i : I) → Quiver.Hom (X i) B
    inst✝ : (CategoryTheory.Presieve.ofArrows X π).hasPullbacks
    ⊢ Iff (∀ (x : (i : I) → P.obj { unop := X i }), CategoryTheory.Presieve.Arrows …
  -/
  simp_rw [← compatible_iff, ← Iso.toEquiv_fun, Equiv.apply_symm_apply]
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    P : CategoryTheory.Functor (Opposite C) (Type w)
    B : C
    I : Type
    X : I → C
    π : (i : I) → Quiver.Hom (X i) B
    inst✝ : (CategoryTheory.Presieve.ofArrows X π).hasPullbacks
    ⊢ Iff (∀ (x : (i : I) → P.obj { unop := X i }), CategoryTheory.Presieve.Arrows …
  -/
  apply forall₂_congr
  /-
    case h
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    P : CategoryTheory.Functor (Opposite C) (Type w)
    B : C
    I : Type
    X : I → C
    π : (i : I) → Quiver.Hom (X i) B
    inst✝ : (CategoryTheory.Presieve.ofArrows X π).hasPullbacks
    ⊢ ∀ (a : (i : I) → P.obj { unop := X i }), CategoryTheory.Presieve.Arrows.Comp …
  -/
  intro x _
  /-
    case h
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    P : CategoryTheory.Functor (Opposite C) (Type w)
    B : C
    I : Type
    X : I → C
    π : (i : I) → Quiver.Hom (X i) B
    inst✝ : (CategoryTheory.Presieve.ofArrows X π).hasPullbacks
    x : (i : I) → P.obj { unop := X i }
    b✝ : CategoryTheory.Presieve.Arrows.Compatible P π x
    ⊢ Iff (ExistsUnique fun t => ∀ (i : I), Eq (P.map (π i).op t) (x i)) (ExistsUn …
  -/
  apply existsUnique_congr
  /-
    case h.h
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    P : CategoryTheory.Functor (Opposite C) (Type w)
    B : C
    I : Type
    X : I → C
    π : (i : I) → Quiver.Hom (X i) B
    inst✝ : (CategoryTheory.Presieve.ofArrows X π).hasPullbacks
    x : (i : I) → P.obj { unop := X i }
    b✝ : CategoryTheory.Presieve.Arrows.Compatible P π x
    ⊢ ∀ (a : P.obj { unop := B }), Iff (∀ (i : I), Eq (P.map (π i).op a) (x i)) (E …
  -/
  intro t
  /-
    case h.h
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    P : CategoryTheory.Functor (Opposite C) (Type w)
    B : C
    I : Type
    X : I → C
    π : (i : I) → Quiver.Hom (X i) B
    inst✝ : (CategoryTheory.Presieve.ofArrows X π).hasPullbacks
    x : (i : I) → P.obj { unop := X i }
    b✝ : CategoryTheory.Presieve.Arrows.Compatible P π x
    t : P.obj { unop := B }
    ⊢ Iff (∀ (i : I), Eq (P.map (π i).op t) (x i)) (Eq (CategoryTheory.Equalizer.P …
  -/
  erw [Equiv.eq_symm_apply]
  /-
    case h.h
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    P : CategoryTheory.Functor (Opposite C) (Type w)
    B : C
    I : Type
    X : I → C
    π : (i : I) → Quiver.Hom (X i) B
    inst✝ : (CategoryTheory.Presieve.ofArrows X π).hasPullbacks
    x : (i : I) → P.obj { unop := X i }
    b✝ : CategoryTheory.Presieve.Arrows.Compatible P π x
    t : P.obj { unop := B }
    ⊢ Iff (∀ (i : I), Eq (P.map (π i).op t) (x i)) (Eq ((CategoryTheory.Limits.Typ …
  -/
  constructor
    /-
      case h.h.mp
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      P : CategoryTheory.Functor (Opposite C) (Type w)
      B : C
      I : Type
      X : I → C
      π : (i : I) → Quiver.Hom (X i) B
      inst✝ : (CategoryTheory.Presieve.ofArrows X π).hasPullbacks
      x : (i : I) → P.obj { unop := X i }
      b✝ : CategoryTheory.Presieve.Arrows.Compatible P π x
      t : P.obj { unop := B }
      ⊢ (∀ (i : I), Eq (P.map (π i).op t) (x i)) → Eq ((CategoryTheory.Limits.Types. …
    -/
  · intro q
    /-
      case h.h.mp
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      P : CategoryTheory.Functor (Opposite C) (Type w)
      B : C
      I : Type
      X : I → C
      π : (i : I) → Quiver.Hom (X i) B
      inst✝ : (CategoryTheory.Presieve.ofArrows X π).hasPullbacks
      x : (i : I) → P.obj { unop := X i }
      b✝ : CategoryTheory.Presieve.Arrows.Compatible P π x
      t : P.obj { unop := B }
      q : ∀ (i : I), Eq (P.map (π i).op t) (x i)
      ⊢ Eq ((CategoryTheory.Limits.Types.productIso fun i => P.obj { unop := X i }). …
    -/
    funext i
    /-
      case h.h.mp.h
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      P : CategoryTheory.Functor (Opposite C) (Type w)
      B : C
      I : Type
      X : I → C
      π : (i : I) → Quiver.Hom (X i) B
      inst✝ : (CategoryTheory.Presieve.ofArrows X π).hasPullbacks
      x : (i : I) → P.obj { unop := X i }
      b✝ : CategoryTheory.Presieve.Arrows.Compatible P π x
      t : P.obj { unop := B }
      q : ∀ (i : I), Eq (P.map (π i).op t) (x i)
      i : I
      ⊢ Eq ((CategoryTheory.Limits.Types.productIso fun i => P.obj { unop := X i }). …
    -/
    simpa [forkMap] using q i
    /-
      🎉 no goals
    -/
    /-
      case h.h.mpr
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      P : CategoryTheory.Functor (Opposite C) (Type w)
      B : C
      I : Type
      X : I → C
      π : (i : I) → Quiver.Hom (X i) B
      inst✝ : (CategoryTheory.Presieve.ofArrows X π).hasPullbacks
      x : (i : I) → P.obj { unop := X i }
      b✝ : CategoryTheory.Presieve.Arrows.Compatible P π x
      t : P.obj { unop := B }
      ⊢ Eq ((CategoryTheory.Limits.Types.productIso fun i => P.obj { unop := X i }). …
    -/
  · intro q i
    /-
      case h.h.mpr
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      P : CategoryTheory.Functor (Opposite C) (Type w)
      B : C
      I : Type
      X : I → C
      π : (i : I) → Quiver.Hom (X i) B
      inst✝ : (CategoryTheory.Presieve.ofArrows X π).hasPullbacks
      x : (i : I) → P.obj { unop := X i }
      b✝ : CategoryTheory.Presieve.Arrows.Compatible P π x
      t : P.obj { unop := B }
      q : Eq ((CategoryTheory.Limits.Types.productIso fun i => P.obj { unop := X i } …
      i : I
      ⊢ Eq (P.map (π i).op t) (x i)
    -/
    rw [← q]
    /-
      case h.h.mpr
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      P : CategoryTheory.Functor (Opposite C) (Type w)
      B : C
      I : Type
      X : I → C
      π : (i : I) → Quiver.Hom (X i) B
      inst✝ : (CategoryTheory.Presieve.ofArrows X π).hasPullbacks
      x : (i : I) → P.obj { unop := X i }
      b✝ : CategoryTheory.Presieve.Arrows.Compatible P π x
      t : P.obj { unop := B }
      q : Eq ((CategoryTheory.Limits.Types.productIso fun i => P.obj { unop := X i } …
      i : I
      ⊢ Eq (P.map (π i).op t) ((CategoryTheory.Limits.Types.productIso fun i => P.ob …
    -/
    simp [forkMap]
    /-
      🎉 no goals
    -/


