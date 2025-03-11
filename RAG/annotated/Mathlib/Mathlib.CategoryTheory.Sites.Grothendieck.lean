/-- The definition of a Grothendieck topology: a set of sieves `J X` on each object `X` satisfying
three axioms:
1. For every object `X`, the maximal sieve is in `J X`.
2. If `S ∈ J X` then its pullback along any `h : Y ⟶ X` is in `J Y`.
3. If `S ∈ J X` and `R` is a sieve on `X`, then provided that the pullback of `R` along any arrow
   `f : Y ⟶ X` in `S` is in `J Y`, we have that `R` itself is in `J X`.

A sieve `S` on `X` is referred to as `J`-covering, (or just covering), if `S ∈ J X`.

See <https://stacks.math.columbia.edu/tag/00Z4>, or [nlab], or [MM92][] Chapter III, Section 2,
Definition 1.
-/
structure GrothendieckTopology where
  /-- A Grothendieck topology on `C` consists of a set of sieves for each object `X`,
    which satisfy some axioms. -/
  sieves : ∀ X : C, Set (Sieve X)
  /-- The sieves associated to each object must contain the top sieve.
    Use `GrothendieckTopology.top_mem`. -/
  top_mem' : ∀ X, ⊤ ∈ sieves X
  /-- Stability under pullback. Use `GrothendieckTopology.pullback_stable`. -/
  pullback_stable' : ∀ ⦃X Y : C⦄ ⦃S : Sieve X⦄ (f : Y ⟶ X), S ∈ sieves X → S.pullback f ∈ sieves Y
  /-- Transitivity of sieves in a Grothendieck topology.
    Use `GrothendieckTopology.transitive`. -/
  transitive' :
    ∀ ⦃X⦄ ⦃S : Sieve X⦄ (_ : S ∈ sieves X) (R : Sieve X),
      (∀ ⦃Y⦄ ⦃f : Y ⟶ X⦄, S f → R.pullback f ∈ sieves Y) → R ∈ sieves X


instance : DFunLike (GrothendieckTopology C) C (fun X ↦ Set (Sieve X)) where
  coe J X := sieves J X
                               /-
                                 C : Type u
                                 inst✝ : CategoryTheory.Category.{v, u} C
                                 J₁ J₂ : CategoryTheory.GrothendieckTopology C
                                 h : Eq ((fun J X => J.sieves X) J₁) ((fun J X => J.sieves X) J₂)
                                 ⊢ Eq J₁ J₂
                               -/
  coe_injective' J₁ J₂ h := by cases J₁; cases J₂; congr
                                                   /-
                                                     🎉 no goals
                                                   -/


/-- An extensionality lemma in terms of the coercion to a pi-type.
We prove this explicitly rather than deriving it so that it is in terms of the coercion rather than
the projection `.sieves`.
-/
@[ext]
theorem ext {J₁ J₂ : GrothendieckTopology C} (h : (J₁ : ∀ X : C, Set (Sieve X)) = J₂) : J₁ = J₂ :=
  DFunLike.coe_injective h


@[simp]
theorem mem_sieves_iff_coe : S ∈ J.sieves X ↔ S ∈ J X :=
  Iff.rfl


/-- Also known as the maximality axiom. -/
@[simp]
theorem top_mem (X : C) : ⊤ ∈ J X :=
  J.top_mem' X


/-- Also known as the stability axiom. -/
@[simp]
theorem pullback_stable (f : Y ⟶ X) (hS : S ∈ J X) : S.pullback f ∈ J Y :=
  J.pullback_stable' f hS


variable {J} in
@[simp]
lemma pullback_mem_iff_of_isIso {i : X ⟶ Y} [IsIso i] {S : Sieve Y} :
    S.pullback i ∈ J _ ↔ S ∈ J _ := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    X Y : C
    J : CategoryTheory.GrothendieckTopology C
    i : Quiver.Hom X Y
    inst✝ : CategoryTheory.IsIso i
    S : CategoryTheory.Sieve Y
    ⊢ Iff (Membership.mem (J X) (CategoryTheory.Sieve.pullback i S)) (Membership.m …
  -/
  refine ⟨fun H ↦ ?_, J.pullback_stable i⟩
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    X Y : C
    J : CategoryTheory.GrothendieckTopology C
    i : Quiver.Hom X Y
    inst✝ : CategoryTheory.IsIso i
    S : CategoryTheory.Sieve Y
    H : Membership.mem (J X) (CategoryTheory.Sieve.pullback i S)
    ⊢ Membership.mem (J Y) S
  -/
  convert J.pullback_stable (inv i) H
  /-
    case h.e'_5
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    X Y : C
    J : CategoryTheory.GrothendieckTopology C
    i : Quiver.Hom X Y
    inst✝ : CategoryTheory.IsIso i
    S : CategoryTheory.Sieve Y
    H : Membership.mem (J X) (CategoryTheory.Sieve.pullback i S)
    ⊢ Eq S (CategoryTheory.Sieve.pullback (CategoryTheory.inv i) (CategoryTheory.S …
  -/
  rw [← Sieve.pullback_comp, IsIso.inv_hom_id, Sieve.pullback_id]
  /-
    🎉 no goals
  -/


theorem transitive (hS : S ∈ J X) (R : Sieve X) (h : ∀ ⦃Y⦄ ⦃f : Y ⟶ X⦄, S f → R.pullback f ∈ J Y) :
    R ∈ J X :=
  J.transitive' hS R h


theorem covering_of_eq_top : S = ⊤ → S ∈ J X := fun h => h.symm ▸ J.top_mem X


/-- If `S` is a subset of `R`, and `S` is covering, then `R` is covering as well.

See <https://stacks.math.columbia.edu/tag/00Z5> (2), or discussion after [MM92] Chapter III,
Section 2, Definition 1.
-/
theorem superset_covering (Hss : S ≤ R) (sjx : S ∈ J X) : R ∈ J X := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X : C
    S R : CategoryTheory.Sieve X
    J : CategoryTheory.GrothendieckTopology C
    Hss : LE.le S R
    sjx : Membership.mem (J X) S
    ⊢ Membership.mem (J X) R
  -/
  apply J.transitive sjx R fun Y f hf => _
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X : C
    S R : CategoryTheory.Sieve X
    J : CategoryTheory.GrothendieckTopology C
    Hss : LE.le S R
    sjx : Membership.mem (J X) S
    ⊢ ∀ (Y : C) (f : Quiver.Hom Y X), S.arrows f → Membership.mem (J Y) (CategoryT …
  -/
  intros Y f hf
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X : C
    S R : CategoryTheory.Sieve X
    J : CategoryTheory.GrothendieckTopology C
    Hss : LE.le S R
    sjx : Membership.mem (J X) S
    Y : C
    f : Quiver.Hom Y X
    hf : S.arrows f
    ⊢ Membership.mem (J Y) (CategoryTheory.Sieve.pullback f R)
  -/
  apply covering_of_eq_top
  /-
    case a
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X : C
    S R : CategoryTheory.Sieve X
    J : CategoryTheory.GrothendieckTopology C
    Hss : LE.le S R
    sjx : Membership.mem (J X) S
    Y : C
    f : Quiver.Hom Y X
    hf : S.arrows f
    ⊢ Eq (CategoryTheory.Sieve.pullback f R) Top.top
  -/
  rw [← top_le_iff, ← S.pullback_eq_top_of_mem hf]
  /-
    case a
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X : C
    S R : CategoryTheory.Sieve X
    J : CategoryTheory.GrothendieckTopology C
    Hss : LE.le S R
    sjx : Membership.mem (J X) S
    Y : C
    f : Quiver.Hom Y X
    hf : S.arrows f
    ⊢ LE.le (CategoryTheory.Sieve.pullback f S) (CategoryTheory.Sieve.pullback f R)
  -/
  apply Sieve.pullback_monotone _ Hss
  /-
    🎉 no goals
  -/


/-- The intersection of two covering sieves is covering.

See <https://stacks.math.columbia.edu/tag/00Z5> (1), or [MM92] Chapter III,
Section 2, Definition 1 (iv).
-/
theorem intersection_covering (rj : R ∈ J X) (sj : S ∈ J X) : R ⊓ S ∈ J X := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X : C
    S R : CategoryTheory.Sieve X
    J : CategoryTheory.GrothendieckTopology C
    rj : Membership.mem (J X) R
    sj : Membership.mem (J X) S
    ⊢ Membership.mem (J X) (Min.min R S)
  -/
  apply J.transitive rj _ fun Y f Hf => _
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X : C
    S R : CategoryTheory.Sieve X
    J : CategoryTheory.GrothendieckTopology C
    rj : Membership.mem (J X) R
    sj : Membership.mem (J X) S
    ⊢ ∀ (Y : C) (f : Quiver.Hom Y X), R.arrows f → Membership.mem (J Y) (CategoryT …
  -/
  intros Y f hf
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X : C
    S R : CategoryTheory.Sieve X
    J : CategoryTheory.GrothendieckTopology C
    rj : Membership.mem (J X) R
    sj : Membership.mem (J X) S
    Y : C
    f : Quiver.Hom Y X
    hf : R.arrows f
    ⊢ Membership.mem (J Y) (CategoryTheory.Sieve.pullback f (Min.min R S))
  -/
  rw [Sieve.pullback_inter, R.pullback_eq_top_of_mem hf]
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X : C
    S R : CategoryTheory.Sieve X
    J : CategoryTheory.GrothendieckTopology C
    rj : Membership.mem (J X) R
    sj : Membership.mem (J X) S
    Y : C
    f : Quiver.Hom Y X
    hf : R.arrows f
    ⊢ Membership.mem (J Y) (Min.min Top.top (CategoryTheory.Sieve.pullback f S))
  -/
  simp [sj]
  /-
    🎉 no goals
  -/


@[simp]
theorem intersection_covering_iff : R ⊓ S ∈ J X ↔ R ∈ J X ∧ S ∈ J X :=
  ⟨fun h => ⟨J.superset_covering inf_le_left h, J.superset_covering inf_le_right h⟩, fun t =>
    intersection_covering _ t.1 t.2⟩


theorem bind_covering {S : Sieve X} {R : ∀ ⦃Y : C⦄ ⦃f : Y ⟶ X⦄, S f → Sieve Y} (hS : S ∈ J X)
    (hR : ∀ ⦃Y⦄ ⦃f : Y ⟶ X⦄ (H : S f), R H ∈ J Y) : Sieve.bind S R ∈ J X :=
  J.transitive hS _ fun _ f hf => superset_covering J (Sieve.le_pullback_bind S R f hf) (hR hf)


/-- The sieve `S` on `X` `J`-covers an arrow `f` to `X` if `S.pullback f ∈ J Y`.
This definition is an alternate way of presenting a Grothendieck topology.
-/
def Covers (S : Sieve X) (f : Y ⟶ X) : Prop :=
  S.pullback f ∈ J Y


theorem covers_iff (S : Sieve X) (f : Y ⟶ X) : J.Covers S f ↔ S.pullback f ∈ J Y :=
  Iff.rfl


                                                                                /-
                                                                                  C : Type u
                                                                                  inst✝ : CategoryTheory.Category.{v, u} C
                                                                                  X : C
                                                                                  J : CategoryTheory.GrothendieckTopology C
                                                                                  S : CategoryTheory.Sieve X
                                                                                  ⊢ Iff (Membership.mem (J X) S) (J.Covers S (CategoryTheory.CategoryStruct.id X))
                                                                                -/
theorem covering_iff_covers_id (S : Sieve X) : S ∈ J X ↔ J.Covers S (𝟙 X) := by simp [covers_iff]
                                                                                /-
                                                                                  🎉 no goals
                                                                                -/


/-- The maximality axiom in 'arrow' form: Any arrow `f` in `S` is covered by `S`. -/
theorem arrow_max (f : Y ⟶ X) (S : Sieve X) (hf : S f) : J.Covers S f := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y : C
    J : CategoryTheory.GrothendieckTopology C
    f : Quiver.Hom Y X
    S : CategoryTheory.Sieve X
    hf : S.arrows f
    ⊢ J.Covers S f
  -/
  rw [Covers, (Sieve.pullback_eq_top_iff_mem f).1 hf]
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y : C
    J : CategoryTheory.GrothendieckTopology C
    f : Quiver.Hom Y X
    S : CategoryTheory.Sieve X
    hf : S.arrows f
    ⊢ Membership.mem (J Y) Top.top
  -/
  apply J.top_mem
  /-
    🎉 no goals
  -/


/-- The stability axiom in 'arrow' form: If `S` covers `f` then `S` covers `g ≫ f` for any `g`. -/
theorem arrow_stable (f : Y ⟶ X) (S : Sieve X) (h : J.Covers S f) {Z : C} (g : Z ⟶ Y) :
    J.Covers S (g ≫ f) := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y : C
    J : CategoryTheory.GrothendieckTopology C
    f : Quiver.Hom Y X
    S : CategoryTheory.Sieve X
    h : J.Covers S f
    Z : C
    g : Quiver.Hom Z Y
    ⊢ J.Covers S (CategoryTheory.CategoryStruct.comp g f)
  -/
  rw [covers_iff] at h ⊢
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y : C
    J : CategoryTheory.GrothendieckTopology C
    f : Quiver.Hom Y X
    S : CategoryTheory.Sieve X
    h : Membership.mem (J Y) (CategoryTheory.Sieve.pullback f S)
    Z : C
    g : Quiver.Hom Z Y
    ⊢ Membership.mem (J Z) (CategoryTheory.Sieve.pullback (CategoryTheory.Category …
  -/
  simp [h, Sieve.pullback_comp]
  /-
    🎉 no goals
  -/


/-- The transitivity axiom in 'arrow' form: If `S` covers `f` and every arrow in `S` is covered by
`R`, then `R` covers `f`.
-/
theorem arrow_trans (f : Y ⟶ X) (S R : Sieve X) (h : J.Covers S f) :
    (∀ {Z : C} (g : Z ⟶ X), S g → J.Covers R g) → J.Covers R f := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y : C
    J : CategoryTheory.GrothendieckTopology C
    f : Quiver.Hom Y X
    S R : CategoryTheory.Sieve X
    h : J.Covers S f
    ⊢ (∀ {Z : C} (g : Quiver.Hom Z X), S.arrows g → J.Covers R g) → J.Covers R f
  -/
  intro k
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y : C
    J : CategoryTheory.GrothendieckTopology C
    f : Quiver.Hom Y X
    S R : CategoryTheory.Sieve X
    h : J.Covers S f
    k : ∀ {Z : C} (g : Quiver.Hom Z X), S.arrows g → J.Covers R g
    ⊢ J.Covers R f
  -/
  apply J.transitive h
  /-
    case h
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y : C
    J : CategoryTheory.GrothendieckTopology C
    f : Quiver.Hom Y X
    S R : CategoryTheory.Sieve X
    h : J.Covers S f
    k : ∀ {Z : C} (g : Quiver.Hom Z X), S.arrows g → J.Covers R g
    ⊢ ∀ ⦃Y_1 : C⦄ ⦃f_1 : Quiver.Hom Y_1 Y⦄, (CategoryTheory.Sieve.pullback f S).ar …
  -/
  intro Z g hg
  /-
    case h
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y : C
    J : CategoryTheory.GrothendieckTopology C
    f : Quiver.Hom Y X
    S R : CategoryTheory.Sieve X
    h : J.Covers S f
    k : ∀ {Z : C} (g : Quiver.Hom Z X), S.arrows g → J.Covers R g
    Z : C
    g : Quiver.Hom Z Y
    hg : (CategoryTheory.Sieve.pullback f S).arrows g
    ⊢ Membership.mem (J Z) (CategoryTheory.Sieve.pullback g (CategoryTheory.Sieve. …
  -/
  rw [← Sieve.pullback_comp]
  /-
    case h
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y : C
    J : CategoryTheory.GrothendieckTopology C
    f : Quiver.Hom Y X
    S R : CategoryTheory.Sieve X
    h : J.Covers S f
    k : ∀ {Z : C} (g : Quiver.Hom Z X), S.arrows g → J.Covers R g
    Z : C
    g : Quiver.Hom Z Y
    hg : (CategoryTheory.Sieve.pullback f S).arrows g
    ⊢ Membership.mem (J Z) (CategoryTheory.Sieve.pullback (CategoryTheory.Category …
  -/
  apply k (g ≫ f) hg
  /-
    🎉 no goals
  -/


theorem arrow_intersect (f : Y ⟶ X) (S R : Sieve X) (hS : J.Covers S f) (hR : J.Covers R f) :
                             /-
                               C : Type u
                               inst✝ : CategoryTheory.Category.{v, u} C
                               X Y : C
                               J : CategoryTheory.GrothendieckTopology C
                               f : Quiver.Hom Y X
                               S R : CategoryTheory.Sieve X
                               hS : J.Covers S f
                               hR : J.Covers R f
                               ⊢ J.Covers (Min.min S R) f
                             -/
    J.Covers (S ⊓ R) f := by simpa [covers_iff] using And.intro hS hR
                             /-
                               🎉 no goals
                             -/


/-- The trivial Grothendieck topology, in which only the maximal sieve is covering. This topology is
also known as the indiscrete, coarse, or chaotic topology.

See [MM92] Chapter III, Section 2, example (a), or
https://en.wikipedia.org/wiki/Grothendieck_topology#The_discrete_and_indiscrete_topologies
-/
def trivial : GrothendieckTopology C where
  sieves _ := {⊤}
  top_mem' _ := rfl
  pullback_stable' X Y S f hf := by
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X✝ Y✝ : C
      S✝ R : CategoryTheory.Sieve X✝
      J : CategoryTheory.GrothendieckTopology C
      X Y : C
      S : CategoryTheory.Sieve X
      f : Quiver.Hom Y X
      hf : Membership.mem ((fun x => Singleton.singleton Top.top) X) S
      ⊢ Membership.mem ((fun x => Singleton.singleton Top.top) Y) (CategoryTheory.Si …
    -/
    rw [Set.mem_singleton_iff] at hf ⊢
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X✝ Y✝ : C
      S✝ R : CategoryTheory.Sieve X✝
      J : CategoryTheory.GrothendieckTopology C
      X Y : C
      S : CategoryTheory.Sieve X
      f : Quiver.Hom Y X
      hf : Eq S Top.top
      ⊢ Eq (CategoryTheory.Sieve.pullback f S) Top.top
    -/
    simp [hf]
    /-
      🎉 no goals
    -/
  transitive' X S hS R hR := by
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X✝ Y : C
      S✝ R✝ : CategoryTheory.Sieve X✝
      J : CategoryTheory.GrothendieckTopology C
      X : C
      S : CategoryTheory.Sieve X
      hS : Membership.mem ((fun x => Singleton.singleton Top.top) X) S
      R : CategoryTheory.Sieve X
      hR : ∀ ⦃Y : C⦄ ⦃f : Quiver.Hom Y X⦄, S.arrows f → Membership.mem ((fun x => Si …
      ⊢ Membership.mem ((fun x => Singleton.singleton Top.top) X) R
    -/
    rw [Set.mem_singleton_iff, ← Sieve.id_mem_iff_eq_top] at hS
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X✝ Y : C
      S✝ R✝ : CategoryTheory.Sieve X✝
      J : CategoryTheory.GrothendieckTopology C
      X : C
      S : CategoryTheory.Sieve X
      hS : S.arrows (CategoryTheory.CategoryStruct.id X)
      R : CategoryTheory.Sieve X
      hR : ∀ ⦃Y : C⦄ ⦃f : Quiver.Hom Y X⦄, S.arrows f → Membership.mem ((fun x => Si …
      ⊢ Membership.mem ((fun x => Singleton.singleton Top.top) X) R
    -/
    simpa using hR hS
    /-
      🎉 no goals
    -/


/-- The discrete Grothendieck topology, in which every sieve is covering.

See https://en.wikipedia.org/wiki/Grothendieck_topology#The_discrete_and_indiscrete_topologies.
-/
def discrete : GrothendieckTopology C where
  sieves _ := Set.univ
                 /-
                   C : Type u
                   inst✝ : CategoryTheory.Category.{v, u} C
                   X Y : C
                   S R : CategoryTheory.Sieve X
                   J : CategoryTheory.GrothendieckTopology C
                   ⊢ ∀ (X : C), Membership.mem ((fun x => Set.univ) X) Top.top
                 -/
  top_mem' := by simp
                 /-
                   🎉 no goals
                 -/
                               /-
                                 C : Type u
                                 inst✝ : CategoryTheory.Category.{v, u} C
                                 X✝ Y✝ : C
                                 S R : CategoryTheory.Sieve X✝
                                 J : CategoryTheory.GrothendieckTopology C
                                 X Y : C
                                 f : CategoryTheory.Sieve X
                                 ⊢ ∀ (f_1 : Quiver.Hom Y X), Membership.mem ((fun x => Set.univ) X) f → Members …
                               -/
  pullback_stable' X Y f := by simp
                               /-
                                 🎉 no goals
                               -/
                    /-
                      C : Type u
                      inst✝ : CategoryTheory.Category.{v, u} C
                      X Y : C
                      S R : CategoryTheory.Sieve X
                      J : CategoryTheory.GrothendieckTopology C
                      ⊢ ∀ ⦃X : C⦄ ⦃S : CategoryTheory.Sieve X⦄, Membership.mem ((fun x => Set.univ)  …
                    -/
  transitive' := by simp
                    /-
                      🎉 no goals
                    -/


theorem trivial_covering : S ∈ trivial C X ↔ S = ⊤ :=
  Set.mem_singleton_iff


/-- See <https://stacks.math.columbia.edu/tag/00Z6> -/
instance instLEGrothendieckTopology : LE (GrothendieckTopology C) where
  le J₁ J₂ := (J₁ : ∀ X : C, Set (Sieve X)) ≤ (J₂ : ∀ X : C, Set (Sieve X))


theorem le_def {J₁ J₂ : GrothendieckTopology C} : J₁ ≤ J₂ ↔ (J₁ : ∀ X : C, Set (Sieve X)) ≤ J₂ :=
  Iff.rfl


/-- See <https://stacks.math.columbia.edu/tag/00Z6> -/
instance : PartialOrder (GrothendieckTopology C) :=
  { instLEGrothendieckTopology with
    le_refl := fun _ => le_def.mpr le_rfl
    le_trans := fun _ _ _ h₁₂ h₂₃ => le_def.mpr (le_trans h₁₂ h₂₃)
    le_antisymm := fun _ _ h₁₂ h₂₁ => GrothendieckTopology.ext (le_antisymm h₁₂ h₂₁) }


/-- See <https://stacks.math.columbia.edu/tag/00Z7> -/
instance : InfSet (GrothendieckTopology C) where
  sInf T :=
    { sieves := sInf (sieves '' T)
      top_mem' := by
        /-
          C : Type u
          inst✝ : CategoryTheory.Category.{v, u} C
          X Y : C
          S R : CategoryTheory.Sieve X
          J : CategoryTheory.GrothendieckTopology C
          T : Set (CategoryTheory.GrothendieckTopology C)
          ⊢ ∀ (X : C), Membership.mem (InfSet.sInf (Set.image CategoryTheory.Grothendiec …
        -/
        rintro X S ⟨⟨_, J, hJ, rfl⟩, rfl⟩
        /-
          case intro.mk.intro.intro
          C : Type u
          inst✝ : CategoryTheory.Category.{v, u} C
          X✝ Y : C
          S R : CategoryTheory.Sieve X✝
          J✝ : CategoryTheory.GrothendieckTopology C
          T : Set (CategoryTheory.GrothendieckTopology C)
          X : C
          J : CategoryTheory.GrothendieckTopology C
          hJ : Membership.mem T J
          ⊢ Membership.mem ((fun f => ↑f X) ⟨J.sieves, ⋯⟩) Top.top
        -/
        simp
        /-
          🎉 no goals
        -/
      pullback_stable' := by
        /-
          C : Type u
          inst✝ : CategoryTheory.Category.{v, u} C
          X Y : C
          S R : CategoryTheory.Sieve X
          J : CategoryTheory.GrothendieckTopology C
          T : Set (CategoryTheory.GrothendieckTopology C)
          ⊢ ∀ ⦃X Y : C⦄ ⦃S : CategoryTheory.Sieve X⦄ (f : Quiver.Hom Y X), Membership.me …
        -/
        rintro X Y S hS f _ ⟨⟨_, J, hJ, rfl⟩, rfl⟩
        /-
          case intro.mk.intro.intro
          C : Type u
          inst✝ : CategoryTheory.Category.{v, u} C
          X✝ Y✝ : C
          S✝ R : CategoryTheory.Sieve X✝
          J✝ : CategoryTheory.GrothendieckTopology C
          T : Set (CategoryTheory.GrothendieckTopology C)
          X Y : C
          S : CategoryTheory.Sieve X
          hS : Quiver.Hom Y X
          f : Membership.mem (InfSet.sInf (Set.image CategoryTheory.GrothendieckTopology …
          J : CategoryTheory.GrothendieckTopology C
          hJ : Membership.mem T J
          ⊢ Membership.mem ((fun f => ↑f Y) ⟨J.sieves, ⋯⟩) (CategoryTheory.Sieve.pullbac …
        -/
        apply J.pullback_stable _ (f _ ⟨⟨_, _, hJ, rfl⟩, rfl⟩)
        /-
          🎉 no goals
        -/
      transitive' := by
        /-
          C : Type u
          inst✝ : CategoryTheory.Category.{v, u} C
          X Y : C
          S R : CategoryTheory.Sieve X
          J : CategoryTheory.GrothendieckTopology C
          T : Set (CategoryTheory.GrothendieckTopology C)
          ⊢ ∀ ⦃X : C⦄ ⦃S : CategoryTheory.Sieve X⦄, Membership.mem (InfSet.sInf (Set.ima …
        -/
        rintro X S hS R h _ ⟨⟨_, J, hJ, rfl⟩, rfl⟩
        apply
          J.transitive (hS _ ⟨⟨_, _, hJ, rfl⟩, rfl⟩) _ fun Y f hf => h hf _ ⟨⟨_, _, hJ, rfl⟩, rfl⟩ }


lemma mem_sInf (s : Set (GrothendieckTopology C)) {X : C} (S : Sieve X) :
    S ∈ sInf s X ↔ ∀ t ∈ s, S ∈ t X := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    s : Set (CategoryTheory.GrothendieckTopology C)
    X : C
    S : CategoryTheory.Sieve X
    ⊢ Iff (Membership.mem ((InfSet.sInf s) X) S) (∀ (t : CategoryTheory.Grothendie …
  -/
  show S ∈ sInf (sieves '' s) X ↔ _
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    s : Set (CategoryTheory.GrothendieckTopology C)
    X : C
    S : CategoryTheory.Sieve X
    ⊢ Iff (Membership.mem (InfSet.sInf (Set.image CategoryTheory.GrothendieckTopol …
  -/
  simp
  /-
    🎉 no goals
  -/


/-- See <https://stacks.math.columbia.edu/tag/00Z7> -/
theorem isGLB_sInf (s : Set (GrothendieckTopology C)) : IsGLB s (sInf s) := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    s : Set (CategoryTheory.GrothendieckTopology C)
    ⊢ IsGLB s (InfSet.sInf s)
  -/
  refine @IsGLB.of_image _ _ _ _ sieves ?_ _ _ ?_
  · #adaptation_note
    /--
    This proof used to be `rfl`,
    but has been temporarily broken by https://github.com/leanprover/lean4/pull/5329.
    It can hopefully be restored after https://github.com/leanprover/lean4/pull/5359
    -/
    /-
      case refine_1
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      s : Set (CategoryTheory.GrothendieckTopology C)
      ⊢ ∀ {x y : CategoryTheory.GrothendieckTopology C}, Iff (LE.le x.sieves y.sieve …
    -/
    exact Iff.rfl
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      s : Set (CategoryTheory.GrothendieckTopology C)
      ⊢ IsGLB (Set.image CategoryTheory.GrothendieckTopology.sieves s) (InfSet.sInf  …
    -/
  · exact _root_.isGLB_sInf _
    /-
      🎉 no goals
    -/


/-- Construct a complete lattice from the `Inf`, but make the trivial and discrete topologies
definitionally equal to the bottom and top respectively.
-/
instance : CompleteLattice (GrothendieckTopology C) :=
  CompleteLattice.copy (completeLatticeOfInf _ isGLB_sInf) _ rfl (discrete C)
    (by
      /-
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        X Y : C
        S R : CategoryTheory.Sieve X
        J : CategoryTheory.GrothendieckTopology C
        ⊢ Eq (CategoryTheory.GrothendieckTopology.discrete C) Top.top
      -/
      apply le_antisymm
        /-
          case a
          C : Type u
          inst✝ : CategoryTheory.Category.{v, u} C
          X Y : C
          S R : CategoryTheory.Sieve X
          J : CategoryTheory.GrothendieckTopology C
          ⊢ LE.le (CategoryTheory.GrothendieckTopology.discrete C) Top.top
        -/
      · exact @CompleteLattice.le_top _ (completeLatticeOfInf _ isGLB_sInf) (discrete C)
        /-
          🎉 no goals
        -/
        /-
          case a
          C : Type u
          inst✝ : CategoryTheory.Category.{v, u} C
          X Y : C
          S R : CategoryTheory.Sieve X
          J : CategoryTheory.GrothendieckTopology C
          ⊢ LE.le Top.top (CategoryTheory.GrothendieckTopology.discrete C)
        -/
      · intro X S _
        /-
          case a
          C : Type u
          inst✝ : CategoryTheory.Category.{v, u} C
          X✝ Y : C
          S✝ R : CategoryTheory.Sieve X✝
          J : CategoryTheory.GrothendieckTopology C
          X : C
          S : CategoryTheory.Sieve X
          a✝ : Membership.mem (Top.top X) S
          ⊢ Membership.mem ((CategoryTheory.GrothendieckTopology.discrete C) X) S
        -/
        apply Set.mem_univ)
        /-
          🎉 no goals
        -/
    (trivial C)
    (by
      /-
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        X Y : C
        S R : CategoryTheory.Sieve X
        J : CategoryTheory.GrothendieckTopology C
        ⊢ Eq (CategoryTheory.GrothendieckTopology.trivial C) Bot.bot
      -/
      apply le_antisymm
        /-
          case a
          C : Type u
          inst✝ : CategoryTheory.Category.{v, u} C
          X Y : C
          S R : CategoryTheory.Sieve X
          J : CategoryTheory.GrothendieckTopology C
          ⊢ LE.le (CategoryTheory.GrothendieckTopology.trivial C) Bot.bot
        -/
      · intro X S hS
        /-
          case a
          C : Type u
          inst✝ : CategoryTheory.Category.{v, u} C
          X✝ Y : C
          S✝ R : CategoryTheory.Sieve X✝
          J : CategoryTheory.GrothendieckTopology C
          X : C
          S : CategoryTheory.Sieve X
          hS : Membership.mem ((CategoryTheory.GrothendieckTopology.trivial C) X) S
          ⊢ Membership.mem (Bot.bot X) S
        -/
        rw [trivial_covering] at hS
        /-
          case a
          C : Type u
          inst✝ : CategoryTheory.Category.{v, u} C
          X✝ Y : C
          S✝ R : CategoryTheory.Sieve X✝
          J : CategoryTheory.GrothendieckTopology C
          X : C
          S : CategoryTheory.Sieve X
          hS : Eq S Top.top
          ⊢ Membership.mem (Bot.bot X) S
        -/
        apply covering_of_eq_top _ hS
        /-
          🎉 no goals
        -/
        /-
          case a
          C : Type u
          inst✝ : CategoryTheory.Category.{v, u} C
          X Y : C
          S R : CategoryTheory.Sieve X
          J : CategoryTheory.GrothendieckTopology C
          ⊢ LE.le Bot.bot (CategoryTheory.GrothendieckTopology.trivial C)
        -/
      · exact @CompleteLattice.bot_le _ (completeLatticeOfInf _ isGLB_sInf) (trivial C))
        /-
          🎉 no goals
        -/
    _ rfl _ rfl _ rfl sInf rfl


instance : Inhabited (GrothendieckTopology C) :=
  ⟨⊤⟩


@[simp]
theorem trivial_eq_bot : trivial C = ⊥ :=
  rfl


@[simp]
theorem discrete_eq_top : discrete C = ⊤ :=
  rfl


@[simp]
theorem bot_covering : S ∈ (⊥ : GrothendieckTopology C) X ↔ S = ⊤ :=
  trivial_covering


@[simp]
theorem top_covering : S ∈ (⊤ : GrothendieckTopology C) X :=
  ⟨⟩


theorem bot_covers (S : Sieve X) (f : Y ⟶ X) : (⊥ : GrothendieckTopology C).Covers S f ↔ S f := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y : C
    S : CategoryTheory.Sieve X
    f : Quiver.Hom Y X
    ⊢ Iff (Bot.bot.Covers S f) (S.arrows f)
  -/
  rw [covers_iff, bot_covering, ← Sieve.pullback_eq_top_iff_mem]
  /-
    🎉 no goals
  -/


@[simp]
theorem top_covers (S : Sieve X) (f : Y ⟶ X) : (⊤ : GrothendieckTopology C).Covers S f := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y : C
    S : CategoryTheory.Sieve X
    f : Quiver.Hom Y X
    ⊢ Top.top.Covers S f
  -/
  simp [covers_iff]
  /-
    🎉 no goals
  -/


/-- The dense Grothendieck topology.

See https://ncatlab.org/nlab/show/dense+topology, or [MM92] Chapter III, Section 2, example (e).
-/
def dense : GrothendieckTopology C where
  sieves X S := ∀ {Y : C} (f : Y ⟶ X), ∃ (Z : _) (g : Z ⟶ Y), S (g ≫ f)
  top_mem' _ Y _ := ⟨Y, 𝟙 Y, ⟨⟩⟩
  pullback_stable' := by
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X Y : C
      S R : CategoryTheory.Sieve X
      J : CategoryTheory.GrothendieckTopology C
      ⊢ ∀ ⦃X Y : C⦄ ⦃S : CategoryTheory.Sieve X⦄ (f : Quiver.Hom Y X), Membership.me …
    -/
    intro X Y S h H Z f
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X✝ Y✝ : C
      S✝ R : CategoryTheory.Sieve X✝
      J : CategoryTheory.GrothendieckTopology C
      X Y : C
      S : CategoryTheory.Sieve X
      h : Quiver.Hom Y X
      H : Membership.mem ((fun X S => ∀ {Y : C} (f : Quiver.Hom Y X), Exists fun Z = …
      Z : C
      f : Quiver.Hom Z Y
      ⊢ Exists fun Z_1 => Exists fun g => (CategoryTheory.Sieve.pullback h S).arrows …
    -/
    rcases H (f ≫ h) with ⟨W, g, H'⟩
    /-
      case intro.intro
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X✝ Y✝ : C
      S✝ R : CategoryTheory.Sieve X✝
      J : CategoryTheory.GrothendieckTopology C
      X Y : C
      S : CategoryTheory.Sieve X
      h : Quiver.Hom Y X
      H : Membership.mem ((fun X S => ∀ {Y : C} (f : Quiver.Hom Y X), Exists fun Z = …
      Z : C
      f : Quiver.Hom Z Y
      W : C
      g : Quiver.Hom W Z
      H' : S.arrows (CategoryTheory.CategoryStruct.comp g (CategoryTheory.CategorySt …
      ⊢ Exists fun Z_1 => Exists fun g => (CategoryTheory.Sieve.pullback h S).arrows …
    -/
    exact ⟨W, g, by simpa⟩
    /-
      🎉 no goals
    -/
  transitive' := by
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X Y : C
      S R : CategoryTheory.Sieve X
      J : CategoryTheory.GrothendieckTopology C
      ⊢ ∀ ⦃X : C⦄ ⦃S : CategoryTheory.Sieve X⦄, Membership.mem ((fun X S => ∀ {Y : C …
    -/
    intro X S H₁ R H₂ Y f
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X✝ Y✝ : C
      S✝ R✝ : CategoryTheory.Sieve X✝
      J : CategoryTheory.GrothendieckTopology C
      X : C
      S : CategoryTheory.Sieve X
      H₁ : Membership.mem ((fun X S => ∀ {Y : C} (f : Quiver.Hom Y X), Exists fun Z  …
      R : CategoryTheory.Sieve X
      H₂ : ∀ ⦃Y : C⦄ ⦃f : Quiver.Hom Y X⦄, S.arrows f → Membership.mem ((fun X S =>  …
      Y : C
      f : Quiver.Hom Y X
      ⊢ Exists fun Z => Exists fun g => R.arrows (CategoryTheory.CategoryStruct.comp …
    -/
    rcases H₁ f with ⟨Z, g, H₃⟩
    /-
      case intro.intro
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X✝ Y✝ : C
      S✝ R✝ : CategoryTheory.Sieve X✝
      J : CategoryTheory.GrothendieckTopology C
      X : C
      S : CategoryTheory.Sieve X
      H₁ : Membership.mem ((fun X S => ∀ {Y : C} (f : Quiver.Hom Y X), Exists fun Z  …
      R : CategoryTheory.Sieve X
      H₂ : ∀ ⦃Y : C⦄ ⦃f : Quiver.Hom Y X⦄, S.arrows f → Membership.mem ((fun X S =>  …
      Y : C
      f : Quiver.Hom Y X
      Z : C
      g : Quiver.Hom Z Y
      H₃ : S.arrows (CategoryTheory.CategoryStruct.comp g f)
      ⊢ Exists fun Z => Exists fun g => R.arrows (CategoryTheory.CategoryStruct.comp …
    -/
    rcases H₂ H₃ (𝟙 Z) with ⟨W, h, H₄⟩
    /-
      case intro.intro.intro.intro
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X✝ Y✝ : C
      S✝ R✝ : CategoryTheory.Sieve X✝
      J : CategoryTheory.GrothendieckTopology C
      X : C
      S : CategoryTheory.Sieve X
      H₁ : Membership.mem ((fun X S => ∀ {Y : C} (f : Quiver.Hom Y X), Exists fun Z  …
      R : CategoryTheory.Sieve X
      H₂ : ∀ ⦃Y : C⦄ ⦃f : Quiver.Hom Y X⦄, S.arrows f → Membership.mem ((fun X S =>  …
      Y : C
      f : Quiver.Hom Y X
      Z : C
      g : Quiver.Hom Z Y
      H₃ : S.arrows (CategoryTheory.CategoryStruct.comp g f)
      W : C
      h : Quiver.Hom W Z
      H₄ : (CategoryTheory.Sieve.pullback (CategoryTheory.CategoryStruct.comp g f) R …
      ⊢ Exists fun Z => Exists fun g => R.arrows (CategoryTheory.CategoryStruct.comp …
    -/
    exact ⟨W, h ≫ g, by simpa using H₄⟩
    /-
      🎉 no goals
    -/


theorem dense_covering : S ∈ dense X ↔ ∀ {Y} (f : Y ⟶ X), ∃ (Z : _) (g : Z ⟶ Y), S (g ≫ f) :=
  Iff.rfl


/--
A category satisfies the right Ore condition if any span can be completed to a commutative square.
NB. Any category with pullbacks obviously satisfies the right Ore condition, see
`right_ore_of_pullbacks`.
-/
def RightOreCondition (C : Type u) [Category.{v} C] : Prop :=
  ∀ {X Y Z : C} (yx : Y ⟶ X) (zx : Z ⟶ X), ∃ (W : _) (wy : W ⟶ Y) (wz : W ⟶ Z), wy ≫ yx = wz ≫ zx


theorem right_ore_of_pullbacks [Limits.HasPullbacks C] : RightOreCondition C := fun _ _ =>
  ⟨_, _, _, Limits.pullback.condition⟩


/-- The atomic Grothendieck topology: a sieve is covering iff it is nonempty.
For the pullback stability condition, we need the right Ore condition to hold.

See https://ncatlab.org/nlab/show/atomic+site, or [MM92] Chapter III, Section 2, example (f).
-/
def atomic (hro : RightOreCondition C) : GrothendieckTopology C where
  sieves X S := ∃ (Y : _) (f : Y ⟶ X), S f
  top_mem' _ := ⟨_, 𝟙 _, ⟨⟩⟩
  pullback_stable' := by
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X Y : C
      S R : CategoryTheory.Sieve X
      J : CategoryTheory.GrothendieckTopology C
      hro : CategoryTheory.GrothendieckTopology.RightOreCondition C
      ⊢ ∀ ⦃X Y : C⦄ ⦃S : CategoryTheory.Sieve X⦄ (f : Quiver.Hom Y X), Membership.me …
    -/
    rintro X Y S h ⟨Z, f, hf⟩
    /-
      case intro.intro
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X✝ Y✝ : C
      S✝ R : CategoryTheory.Sieve X✝
      J : CategoryTheory.GrothendieckTopology C
      hro : CategoryTheory.GrothendieckTopology.RightOreCondition C
      X Y : C
      S : CategoryTheory.Sieve X
      h : Quiver.Hom Y X
      Z : C
      f : Quiver.Hom Z X
      hf : S.arrows f
      ⊢ Membership.mem ((fun X S => Exists fun Y => Exists fun f => S.arrows f) Y) ( …
    -/
    rcases hro h f with ⟨W, g, k, comm⟩
    /-
      case intro.intro.intro.intro.intro
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X✝ Y✝ : C
      S✝ R : CategoryTheory.Sieve X✝
      J : CategoryTheory.GrothendieckTopology C
      hro : CategoryTheory.GrothendieckTopology.RightOreCondition C
      X Y : C
      S : CategoryTheory.Sieve X
      h : Quiver.Hom Y X
      Z : C
      f : Quiver.Hom Z X
      hf : S.arrows f
      W : C
      g : Quiver.Hom W Y
      k : Quiver.Hom W Z
      comm : Eq (CategoryTheory.CategoryStruct.comp g h) (CategoryTheory.CategoryStr …
      ⊢ Membership.mem ((fun X S => Exists fun Y => Exists fun f => S.arrows f) Y) ( …
    -/
    refine ⟨_, g, ?_⟩
    /-
      case intro.intro.intro.intro.intro
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X✝ Y✝ : C
      S✝ R : CategoryTheory.Sieve X✝
      J : CategoryTheory.GrothendieckTopology C
      hro : CategoryTheory.GrothendieckTopology.RightOreCondition C
      X Y : C
      S : CategoryTheory.Sieve X
      h : Quiver.Hom Y X
      Z : C
      f : Quiver.Hom Z X
      hf : S.arrows f
      W : C
      g : Quiver.Hom W Y
      k : Quiver.Hom W Z
      comm : Eq (CategoryTheory.CategoryStruct.comp g h) (CategoryTheory.CategoryStr …
      ⊢ (CategoryTheory.Sieve.pullback h S).arrows g
    -/
    simp [comm, hf]
    /-
      🎉 no goals
    -/
  transitive' := by
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X Y : C
      S R : CategoryTheory.Sieve X
      J : CategoryTheory.GrothendieckTopology C
      hro : CategoryTheory.GrothendieckTopology.RightOreCondition C
      ⊢ ∀ ⦃X : C⦄ ⦃S : CategoryTheory.Sieve X⦄, Membership.mem ((fun X S => Exists f …
    -/
    rintro X S ⟨Y, f, hf⟩ R h
    /-
      case intro.intro
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X✝ Y✝ : C
      S✝ R✝ : CategoryTheory.Sieve X✝
      J : CategoryTheory.GrothendieckTopology C
      hro : CategoryTheory.GrothendieckTopology.RightOreCondition C
      X : C
      S : CategoryTheory.Sieve X
      Y : C
      f : Quiver.Hom Y X
      hf : S.arrows f
      R : CategoryTheory.Sieve X
      h : ∀ ⦃Y : C⦄ ⦃f : Quiver.Hom Y X⦄, S.arrows f → Membership.mem ((fun X S => E …
      ⊢ Membership.mem ((fun X S => Exists fun Y => Exists fun f => S.arrows f) X) R
    -/
    rcases h hf with ⟨Z, g, hg⟩
    /-
      case intro.intro.intro.intro
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X✝ Y✝ : C
      S✝ R✝ : CategoryTheory.Sieve X✝
      J : CategoryTheory.GrothendieckTopology C
      hro : CategoryTheory.GrothendieckTopology.RightOreCondition C
      X : C
      S : CategoryTheory.Sieve X
      Y : C
      f : Quiver.Hom Y X
      hf : S.arrows f
      R : CategoryTheory.Sieve X
      h : ∀ ⦃Y : C⦄ ⦃f : Quiver.Hom Y X⦄, S.arrows f → Membership.mem ((fun X S => E …
      Z : C
      g : Quiver.Hom Z Y
      hg : (CategoryTheory.Sieve.pullback f R).arrows g
      ⊢ Membership.mem ((fun X S => Exists fun Y => Exists fun f => S.arrows f) X) R
    -/
    exact ⟨_, _, hg⟩
    /-
      🎉 no goals
    -/



/-- `J.Cover X` denotes the poset of covers of `X` with respect to the
Grothendieck topology `J`. -/
-- Porting note: Lean 3 inferred `Type max u v`, Lean 4 by default gives `Type (max 0 u v)`
def Cover (X : C) : Type max u v :=
  { S : Sieve X // S ∈ J X } -- deriving Preorder

-- Porting note: `deriving` didn't work above, so we add the preorder instance manually.

instance (X : C) : Preorder (J.Cover X) :=
  show Preorder {S : Sieve X // S ∈ J X} from inferInstance


instance : CoeOut (J.Cover X) (Sieve X) := ⟨fun S => S.1⟩


instance : CoeFun (J.Cover X) fun _ => ∀ ⦃Y⦄ (_ : Y ⟶ X), Prop := ⟨fun S => (S : Sieve X)⟩


theorem condition (S : J.Cover X) : (S : Sieve X) ∈ J X := S.2


@[ext]
theorem ext (S T : J.Cover X) (h : ∀ ⦃Y⦄ (f : Y ⟶ X), S f ↔ T f) : S = T :=
  Subtype.ext <| Sieve.ext h


instance : OrderTop (J.Cover X) :=
  { (inferInstance : Preorder (J.Cover X)) with
    top := ⟨⊤, J.top_mem _⟩
                                /-
                                  C : Type u
                                  inst✝ : CategoryTheory.Category.{v, u} C
                                  X Y : C
                                  S R : CategoryTheory.Sieve X
                                  J : CategoryTheory.GrothendieckTopology C
                                  x✝³ : J.Cover X
                                  x✝² : C
                                  x✝¹ : Quiver.Hom x✝² X
                                  x✝ : ((fun a => ↑a) x✝³).arrows x✝¹
                                  ⊢ ((fun a => ↑a) Top.top).arrows x✝¹
                                -/
    le_top := fun _ _ _ _ => by tauto }
                                /-
                                  🎉 no goals
                                -/


instance : SemilatticeInf (J.Cover X) :=
  { (inferInstance : Preorder _) with
    inf := fun S T => ⟨S ⊓ T, J.intersection_covering S.condition T.condition⟩
                                                             /-
                                                               C : Type u
                                                               inst✝ : CategoryTheory.Category.{v, u} C
                                                               X Y✝ : C
                                                               S R : CategoryTheory.Sieve X
                                                               J : CategoryTheory.GrothendieckTopology C
                                                               x✝¹ x✝ : J.Cover X
                                                               h1 : LE.le x✝¹ x✝
                                                               h2 : LE.le x✝ x✝¹
                                                               Y : C
                                                               f : Quiver.Hom Y X
                                                               ⊢ (↑x✝¹).arrows f → (↑x✝).arrows f
                                                             -/
                                                             /-
                                                               🎉 no goals
                                                             -/
    le_antisymm := fun _ _ h1 h2 => ext _ _ fun {Y} f => ⟨by apply h1, by apply h2⟩
                                                                          /-
                                                                            🎉 no goals
                                                                          -/
    inf_le_left := fun _ _ _ _ hf => hf.1
    inf_le_right := fun _ _ _ _ hf => hf.2
    le_inf := fun _ _ _ h1 h2 _ _ h => ⟨h1 _ h, h2 _ h⟩ }


instance : Inhabited (J.Cover X) :=
  ⟨⊤⟩


/-- An auxiliary structure, used to define `S.index`. -/
-- Porting note (https://github.com/leanprover-community/mathlib4/issues/5171): this linter isn't ported yet.
-- @[nolint has_nonempty_instance]
@[ext]
structure Arrow (S : J.Cover X) where
  /-- The source of the arrow. -/
  Y : C
  /-- The arrow itself. -/
  f : Y ⟶ X
  /-- The given arrow is contained in the given sieve. -/
  hf : S f


/-- Relation between two elements in `S.arrow`, the data of which
involves a commutative square. -/
@[ext]
structure Arrow.Relation {S : J.Cover X} (I₁ I₂ : S.Arrow) where
  /-- The source of the arrows defining the relation. -/
  Z : C
  /-- The first arrow defining the relation. -/
  g₁ : Z ⟶ I₁.Y
  /-- The second arrow defining the relation. -/
  g₂ : Z ⟶ I₂.Y
  /-- The relation itself. -/
  w : g₁ ≫ I₁.f = g₂ ≫ I₂.f := by aesop_cat


attribute [reassoc] Arrow.Relation.w


/-- Given `I : S.Arrow` and a morphism `g : Z ⟶ I.Y`, this is the arrow in `S.Arrow`
corresponding to `g ≫ I.f`. -/
@[simps]
def Arrow.precomp {S : J.Cover X} (I : S.Arrow) {Z : C} (g : Z ⟶ I.Y) : S.Arrow :=
  ⟨Z, g ≫ I.f, S.1.downward_closed I.hf g⟩


/-- Given `I : S.Arrow` and a morphism `g : Z ⟶ I.Y`, this is the obvious relation
from `I.precomp g` to `I`. -/
@[simps]
def Arrow.precompRelation {S : J.Cover X} (I : S.Arrow) {Z : C} (g : Z ⟶ I.Y) :
    (I.precomp g).Relation I where
  g₁ := 𝟙 _
  g₂ := g


/-- Map an `Arrow` along a refinement `S ⟶ T`. -/
@[simps]
def Arrow.map {S T : J.Cover X} (I : S.Arrow) (f : S ⟶ T) : T.Arrow :=
  ⟨I.Y, I.f, f.le _ I.hf⟩


/-- Map an `Arrow.Relation` along a refinement `S ⟶ T`. -/
@[simps]
def Arrow.Relation.map {S T : J.Cover X} {I₁ I₂ : S.Arrow}
    (r : I₁.Relation I₂) (f : S ⟶ T) : (I₁.map f).Relation (I₂.map f) where
  w := r.w


/-- Pull back a cover along a morphism. -/
def pullback (S : J.Cover X) (f : Y ⟶ X) : J.Cover Y :=
  ⟨Sieve.pullback f S, J.pullback_stable _ S.condition⟩


/-- An arrow of `S.pullback f` gives rise to an arrow of `S`. -/
@[simps]
def Arrow.base {f : Y ⟶ X} {S : J.Cover X} (I : (S.pullback f).Arrow) : S.Arrow :=
  ⟨I.Y, I.f ≫ f, I.hf⟩


/-- A relation of `S.pullback f` gives rise to a relation of `S`. -/
def Arrow.Relation.base
    {f : Y ⟶ X} {S : J.Cover X} {I₁ I₂ : (S.pullback f).Arrow}
    (r : I₁.Relation I₂) : I₁.base.Relation I₂.base where
  g₁ := r.g₁
  g₂ := r.g₂
          /-
            C : Type u
            inst✝ : CategoryTheory.Category.{v, u} C
            X Y : C
            S✝ R : CategoryTheory.Sieve X
            J : CategoryTheory.GrothendieckTopology C
            f : Quiver.Hom Y X
            S : J.Cover X
            I₁ I₂ : (S.pullback f).Arrow
            r : I₁.Relation I₂
            ⊢ Eq (CategoryTheory.CategoryStruct.comp r.g₁ I₁.base.f) (CategoryTheory.Categ …
          -/
  w := by simp [r.w_assoc]
          /-
            🎉 no goals
          -/


@[simp]
theorem coe_pullback {Z : C} (f : Y ⟶ X) (g : Z ⟶ Y) (S : J.Cover X) :
    (S.pullback f) g ↔ S (g ≫ f) :=
  Iff.rfl


/-- The isomorphism between `S` and the pullback of `S` w.r.t. the identity. -/
def pullbackId (S : J.Cover X) : S.pullback (𝟙 X) ≅ S :=
                                         /-
                                           C : Type u
                                           inst✝ : CategoryTheory.Category.{v, u} C
                                           X Y✝ : C
                                           S✝ R : CategoryTheory.Sieve X
                                           J : CategoryTheory.GrothendieckTopology C
                                           S : J.Cover X
                                           Y : C
                                           f : Quiver.Hom Y X
                                           ⊢ Iff ((↑(S.pullback (CategoryTheory.CategoryStruct.id X))).arrows f) ((↑S).ar …
                                         -/
  eqToIso <| Cover.ext _ _ fun Y f => by simp
                                         /-
                                           🎉 no goals
                                         -/


/-- Pulling back with respect to a composition is the composition of the pullbacks. -/
def pullbackComp {X Y Z : C} (S : J.Cover X) (f : Z ⟶ Y) (g : Y ⟶ X) :
    S.pullback (f ≫ g) ≅ (S.pullback g).pullback f :=
                                         /-
                                           C : Type u
                                           inst✝ : CategoryTheory.Category.{v, u} C
                                           X✝ Y✝¹ : C
                                           S✝ R : CategoryTheory.Sieve X✝
                                           J : CategoryTheory.GrothendieckTopology C
                                           X Y✝ Z : C
                                           S : J.Cover X
                                           f✝ : Quiver.Hom Z Y✝
                                           g : Quiver.Hom Y✝ X
                                           Y : C
                                           f : Quiver.Hom Y Z
                                           ⊢ Iff ((↑(S.pullback (CategoryTheory.CategoryStruct.comp f✝ g))).arrows f) ((↑ …
                                         -/
  eqToIso <| Cover.ext _ _ fun Y f => by simp
                                         /-
                                           🎉 no goals
                                         -/


/-- Combine a family of covers over a cover. -/
def bind {X : C} (S : J.Cover X) (T : ∀ I : S.Arrow, J.Cover I.Y) : J.Cover X :=
  ⟨Sieve.bind S fun Y f hf => T ⟨Y, f, hf⟩,
    J.bind_covering S.condition fun _ _ _ => (T { Y := _, f := _, hf := _ }).condition⟩


/-- The canonical morphism from `S.bind T` to `T`. -/
def bindToBase {X : C} (S : J.Cover X) (T : ∀ I : S.Arrow, J.Cover I.Y) : S.bind T ⟶ S :=
  homOfLE <| by
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X✝ Y : C
      S✝ R : CategoryTheory.Sieve X✝
      J : CategoryTheory.GrothendieckTopology C
      X : C
      S : J.Cover X
      T : (I : S.Arrow) → J.Cover I.Y
      ⊢ LE.le (S.bind T) S
    -/
    rintro Y f ⟨Z, e1, e2, h1, _, h3⟩
    /-
      case intro.intro.intro.intro.intro
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X✝ Y✝ : C
      S✝ R : CategoryTheory.Sieve X✝
      J : CategoryTheory.GrothendieckTopology C
      X : C
      S : J.Cover X
      T : (I : S.Arrow) → J.Cover I.Y
      Y : C
      f : Quiver.Hom Y X
      Z : C
      e1 : Quiver.Hom Y Z
      e2 : Quiver.Hom Z X
      h1 : (↑S).arrows e2
      left✝ : ((fun Y f hf => ↑(T { Y := Y, f := f, hf := hf })) Z e2 h1).arrows e1
      h3 : Eq (CategoryTheory.CategoryStruct.comp e1 e2) f
      ⊢ ((fun a => ↑a) S).arrows f
    -/
    rw [← h3]
    /-
      case intro.intro.intro.intro.intro
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X✝ Y✝ : C
      S✝ R : CategoryTheory.Sieve X✝
      J : CategoryTheory.GrothendieckTopology C
      X : C
      S : J.Cover X
      T : (I : S.Arrow) → J.Cover I.Y
      Y : C
      f : Quiver.Hom Y X
      Z : C
      e1 : Quiver.Hom Y Z
      e2 : Quiver.Hom Z X
      h1 : (↑S).arrows e2
      left✝ : ((fun Y f hf => ↑(T { Y := Y, f := f, hf := hf })) Z e2 h1).arrows e1
      h3 : Eq (CategoryTheory.CategoryStruct.comp e1 e2) f
      ⊢ ((fun a => ↑a) S).arrows (CategoryTheory.CategoryStruct.comp e1 e2)
    -/
    apply Sieve.downward_closed
    /-
      case intro.intro.intro.intro.intro.x
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X✝ Y✝ : C
      S✝ R : CategoryTheory.Sieve X✝
      J : CategoryTheory.GrothendieckTopology C
      X : C
      S : J.Cover X
      T : (I : S.Arrow) → J.Cover I.Y
      Y : C
      f : Quiver.Hom Y X
      Z : C
      e1 : Quiver.Hom Y Z
      e2 : Quiver.Hom Z X
      h1 : (↑S).arrows e2
      left✝ : ((fun Y f hf => ↑(T { Y := Y, f := f, hf := hf })) Z e2 h1).arrows e1
      h3 : Eq (CategoryTheory.CategoryStruct.comp e1 e2) f
      ⊢ ((fun a => ↑a) S).arrows e2
    -/
    exact h1
    /-
      🎉 no goals
    -/


/-- An arrow in bind has the form `A ⟶ B ⟶ X` where `A ⟶ B` is an arrow in `T I` for some `I`.
 and `B ⟶ X` is an arrow of `S`. This is the object `B`. -/
noncomputable def Arrow.middle {X : C} {S : J.Cover X} {T : ∀ I : S.Arrow, J.Cover I.Y}
    (I : (S.bind T).Arrow) : C :=
  I.hf.choose


/-- An arrow in bind has the form `A ⟶ B ⟶ X` where `A ⟶ B` is an arrow in `T I` for some `I`.
 and `B ⟶ X` is an arrow of `S`. This is the hom `A ⟶ B`. -/
noncomputable def Arrow.toMiddleHom {X : C} {S : J.Cover X} {T : ∀ I : S.Arrow, J.Cover I.Y}
    (I : (S.bind T).Arrow) : I.Y ⟶ I.middle :=
  I.hf.choose_spec.choose


/-- An arrow in bind has the form `A ⟶ B ⟶ X` where `A ⟶ B` is an arrow in `T I` for some `I`.
 and `B ⟶ X` is an arrow of `S`. This is the hom `B ⟶ X`. -/
noncomputable def Arrow.fromMiddleHom {X : C} {S : J.Cover X} {T : ∀ I : S.Arrow, J.Cover I.Y}
    (I : (S.bind T).Arrow) : I.middle ⟶ X :=
  I.hf.choose_spec.choose_spec.choose


theorem Arrow.from_middle_condition {X : C} {S : J.Cover X} {T : ∀ I : S.Arrow, J.Cover I.Y}
    (I : (S.bind T).Arrow) : S I.fromMiddleHom :=
  I.hf.choose_spec.choose_spec.choose_spec.choose


/-- An arrow in bind has the form `A ⟶ B ⟶ X` where `A ⟶ B` is an arrow in `T I` for some `I`.
 and `B ⟶ X` is an arrow of `S`. This is the hom `B ⟶ X`, as an arrow. -/
noncomputable def Arrow.fromMiddle {X : C} {S : J.Cover X} {T : ∀ I : S.Arrow, J.Cover I.Y}
    (I : (S.bind T).Arrow) : S.Arrow :=
  ⟨_, I.fromMiddleHom, I.from_middle_condition⟩


theorem Arrow.to_middle_condition {X : C} {S : J.Cover X} {T : ∀ I : S.Arrow, J.Cover I.Y}
    (I : (S.bind T).Arrow) : (T I.fromMiddle) I.toMiddleHom :=
  I.hf.choose_spec.choose_spec.choose_spec.choose_spec.1


/-- An arrow in bind has the form `A ⟶ B ⟶ X` where `A ⟶ B` is an arrow in `T I` for some `I`.
 and `B ⟶ X` is an arrow of `S`. This is the hom `A ⟶ B`, as an arrow. -/
noncomputable def Arrow.toMiddle {X : C} {S : J.Cover X} {T : ∀ I : S.Arrow, J.Cover I.Y}
    (I : (S.bind T).Arrow) : (T I.fromMiddle).Arrow :=
  ⟨_, I.toMiddleHom, I.to_middle_condition⟩


theorem Arrow.middle_spec {X : C} {S : J.Cover X} {T : ∀ I : S.Arrow, J.Cover I.Y}
    (I : (S.bind T).Arrow) : I.toMiddleHom ≫ I.fromMiddleHom = I.f :=
  I.hf.choose_spec.choose_spec.choose_spec.choose_spec.2


/-- An auxiliary structure, used to define `S.index`. -/
-- Porting note (https://github.com/leanprover-community/mathlib4/issues/5171): this linter isn't ported yet.
-- @[nolint has_nonempty_instance, ext]
@[ext]
structure Relation (S : J.Cover X) where
  /-- The first arrow. -/
  fst : S.Arrow
  /-- The second arrow. -/
  snd : S.Arrow
  /-- The relation between the two arrows. -/
  r : fst.Relation snd


/-- Constructor for `Cover.Relation` which takes as an input
`r : I₁.Relation I₂` with `I₁ I₂ : S.Arrow`. -/
@[simps]
def Relation.mk' {S : J.Cover X} {fst snd : S.Arrow} (r : fst.Relation snd) :
    S.Relation where
  r := r

-- This is used extensively in `Plus.lean`, etc.
-- We place this definition here as it will be used in `Sheaf.lean` as well.

/-- To every `S : J.Cover X` and presheaf `P`, associate a `MulticospanIndex`. -/
@[simps]
def index {D : Type u₁} [Category.{v₁} D] (S : J.Cover X) (P : Cᵒᵖ ⥤ D) :
    Limits.MulticospanIndex D where
  L := S.Arrow
  R := S.Relation
  fstTo I := I.fst
  sndTo I := I.snd
  left I := P.obj (Opposite.op I.Y)
  right I := P.obj (Opposite.op I.r.Z)
  fst I := P.map I.r.g₁.op
  snd I := P.map I.r.g₂.op


/-- The natural multifork associated to `S : J.Cover X` for a presheaf `P`.
Saying that this multifork is a limit is essentially equivalent to the sheaf condition at the
given object for the given covering sieve. See `Sheaf.lean` for an equivalent sheaf condition
using this.
-/
abbrev multifork {D : Type u₁} [Category.{v₁} D] (S : J.Cover X) (P : Cᵒᵖ ⥤ D) :
    Limits.Multifork (S.index P) :=
  Limits.Multifork.ofι _ (P.obj (Opposite.op X)) (fun I => P.map I.f.op)
    (by
      /-
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        X Y : C
        S✝ R : CategoryTheory.Sieve X
        J : CategoryTheory.GrothendieckTopology C
        D : Type u₁
        inst✝ : CategoryTheory.Category.{v₁, u₁} D
        S : J.Cover X
        P : CategoryTheory.Functor (Opposite C) D
        ⊢ ∀ (b : (S.index P).R), Eq (CategoryTheory.CategoryStruct.comp ((fun I => P.m …
      -/
      intro I
      /-
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        X Y : C
        S✝ R : CategoryTheory.Sieve X
        J : CategoryTheory.GrothendieckTopology C
        D : Type u₁
        inst✝ : CategoryTheory.Category.{v₁, u₁} D
        S : J.Cover X
        P : CategoryTheory.Functor (Opposite C) D
        I : (S.index P).R
        ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun I => P.map I.f.op) ((S.index P) …
      -/
      dsimp
      /-
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        X Y : C
        S✝ R : CategoryTheory.Sieve X
        J : CategoryTheory.GrothendieckTopology C
        D : Type u₁
        inst✝ : CategoryTheory.Category.{v₁, u₁} D
        S : J.Cover X
        P : CategoryTheory.Functor (Opposite C) D
        I : (S.index P).R
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (P.map I.fst.f.op) (P.map I.r.g₁.op)) …
      -/
      simp only [← P.map_comp, ← op_comp, I.r.w])
      /-
        🎉 no goals
      -/


/-- The canonical map from `P.obj (op X)` to the multiequalizer associated to a covering sieve,
assuming such a multiequalizer exists. This will be used in `Sheaf.lean` to provide an equivalent
sheaf condition in terms of multiequalizers. -/
noncomputable abbrev toMultiequalizer {D : Type u₁} [Category.{v₁} D] (S : J.Cover X)
    (P : Cᵒᵖ ⥤ D) [Limits.HasMultiequalizer (S.index P)] :
    P.obj (Opposite.op X) ⟶ Limits.multiequalizer (S.index P) :=
  Limits.Multiequalizer.lift _ _ (fun I => P.map I.f.op)
    (by
      /-
        C : Type u
        inst✝² : CategoryTheory.Category.{v, u} C
        X Y : C
        S✝ R : CategoryTheory.Sieve X
        J : CategoryTheory.GrothendieckTopology C
        D : Type u₁
        inst✝¹ : CategoryTheory.Category.{v₁, u₁} D
        S : J.Cover X
        P : CategoryTheory.Functor (Opposite C) D
        inst✝ : CategoryTheory.Limits.HasMultiequalizer (S.index P)
        ⊢ ∀ (b : (S.index P).R), Eq (CategoryTheory.CategoryStruct.comp ((fun I => P.m …
      -/
      intro I
      /-
        C : Type u
        inst✝² : CategoryTheory.Category.{v, u} C
        X Y : C
        S✝ R : CategoryTheory.Sieve X
        J : CategoryTheory.GrothendieckTopology C
        D : Type u₁
        inst✝¹ : CategoryTheory.Category.{v₁, u₁} D
        S : J.Cover X
        P : CategoryTheory.Functor (Opposite C) D
        inst✝ : CategoryTheory.Limits.HasMultiequalizer (S.index P)
        I : (S.index P).R
        ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun I => P.map I.f.op) ((S.index P) …
      -/
      dsimp only [index, Relation.fst, Relation.snd]
      /-
        C : Type u
        inst✝² : CategoryTheory.Category.{v, u} C
        X Y : C
        S✝ R : CategoryTheory.Sieve X
        J : CategoryTheory.GrothendieckTopology C
        D : Type u₁
        inst✝¹ : CategoryTheory.Category.{v₁, u₁} D
        S : J.Cover X
        P : CategoryTheory.Functor (Opposite C) D
        inst✝ : CategoryTheory.Limits.HasMultiequalizer (S.index P)
        I : (S.index P).R
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (P.map I.fst.f.op) (P.map I.r.g₁.op)) …
      -/
      simp only [← P.map_comp, ← op_comp, I.r.w])
      /-
        🎉 no goals
      -/


/-- Pull back a cover along a morphism. -/
@[simps obj]
def pullback (f : Y ⟶ X) : J.Cover X ⥤ J.Cover Y where
  obj S := S.pullback f
  map f := (Sieve.pullback_monotone _ f.le).hom


/-- Pulling back along the identity is naturally isomorphic to the identity functor. -/
def pullbackId (X : C) : J.pullback (𝟙 X) ≅ 𝟭 _ :=
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X✝ Y : C
    S R : CategoryTheory.Sieve X✝
    J : CategoryTheory.GrothendieckTopology C
    X : C
    ⊢ ∀ {X_1 Y : J.Cover X} (f : Quiver.Hom X_1 Y), Eq (CategoryTheory.CategoryStr …
  -/
  NatIso.ofComponents fun S => S.pullbackId
  /-
    🎉 no goals
  -/


/-- Pulling back along a composition is naturally isomorphic to
the composition of the pullbacks. -/
def pullbackComp {X Y Z : C} (f : X ⟶ Y) (g : Y ⟶ Z) :
    J.pullback (f ≫ g) ≅ J.pullback g ⋙ J.pullback f :=
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X✝ Y✝ : C
    S R : CategoryTheory.Sieve X✝
    J : CategoryTheory.GrothendieckTopology C
    X Y Z : C
    f : Quiver.Hom X Y
    g : Quiver.Hom Y Z
    ⊢ ∀ {X_1 Y_1 : J.Cover Z} (f_1 : Quiver.Hom X_1 Y_1), Eq (CategoryTheory.Categ …
  -/
  NatIso.ofComponents fun S => S.pullbackComp f g
  /-
    🎉 no goals
  -/


