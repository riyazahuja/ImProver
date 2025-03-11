/--
A (Grothendieck) pretopology on `C` consists of a collection of families of morphisms with a fixed
target `X` for every object `X` in `C`, called "coverings" of `X`, which satisfies the following
three axioms:
1. Every family consisting of a single isomorphism is a covering family.
2. The collection of covering families is stable under pullback.
3. Given a covering family, and a covering family on each domain of the former, the composition
   is a covering family.

In some sense, a pretopology can be seen as Grothendieck topology with weaker saturation conditions,
in that each covering is not necessarily downward closed.

See: https://ncatlab.org/nlab/show/Grothendieck+pretopology, or
https://stacks.math.columbia.edu/tag/00VH, or [MM92] Chapter III, Section 2, Definition 2.
Note that Stacks calls a category together with a pretopology a site, and [MM92] calls this
a basis for a topology.
-/
@[ext]
structure Pretopology where
  coverings : ∀ X : C, Set (Presieve X)
  has_isos : ∀ ⦃X Y⦄ (f : Y ⟶ X) [IsIso f], Presieve.singleton f ∈ coverings X
  pullbacks : ∀ ⦃X Y⦄ (f : Y ⟶ X) (S), S ∈ coverings X → pullbackArrows f S ∈ coverings Y
  transitive :
    ∀ ⦃X : C⦄ (S : Presieve X) (Ti : ∀ ⦃Y⦄ (f : Y ⟶ X), S f → Presieve Y),
      S ∈ coverings X → (∀ ⦃Y⦄ (f) (H : S f), Ti f H ∈ coverings Y) → S.bind Ti ∈ coverings X


instance : CoeFun (Pretopology C) fun _ => ∀ X : C, Set (Presieve X) :=
  ⟨coverings⟩


instance LE : LE (Pretopology C) where
  le K₁ K₂ := (K₁ : ∀ X : C, Set (Presieve X)) ≤ K₂


theorem le_def {K₁ K₂ : Pretopology C} : K₁ ≤ K₂ ↔ (K₁ : ∀ X : C, Set (Presieve X)) ≤ K₂ :=
  Iff.rfl


instance : PartialOrder (Pretopology C) :=
  { Pretopology.LE with
    le_refl := fun _ => le_def.mpr le_rfl
    le_trans := fun _ _ _ h₁₂ h₂₃ => le_def.mpr (le_trans h₁₂ h₂₃)
    le_antisymm := fun _ _ h₁₂ h₂₁ => Pretopology.ext (le_antisymm h₁₂ h₂₁) }


instance orderTop : OrderTop (Pretopology C) where
  top :=
    { coverings := fun _ => Set.univ
      has_isos := fun _ _ _ _ => Set.mem_univ _
      pullbacks := fun _ _ _ _ _ => Set.mem_univ _
      transitive := fun _ _ _ _ _ => Set.mem_univ _ }
  le_top _ _ _ _ := Set.mem_univ _


instance : Inhabited (Pretopology C) :=
  ⟨⊤⟩


/-- A pretopology `K` can be completed to a Grothendieck topology `J` by declaring a sieve to be
`J`-covering if it contains a family in `K`.

See <https://stacks.math.columbia.edu/tag/00ZC>, or [MM92] Chapter III, Section 2, Equation (2).
-/
def toGrothendieck (K : Pretopology C) : GrothendieckTopology C where
  sieves X S := ∃ R ∈ K X, R ≤ (S : Presieve _)
  top_mem' _ := ⟨Presieve.singleton (𝟙 _), K.has_isos _, fun _ _ _ => ⟨⟩⟩
  pullback_stable' X Y S g := by
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.Limits.HasPullbacks C
      K : CategoryTheory.Pretopology C
      X Y : C
      S : CategoryTheory.Sieve X
      g : Quiver.Hom Y X
      ⊢ Membership.mem ((fun X S => Exists fun R => And (Membership.mem (K.coverings …
    -/
    rintro ⟨R, hR, RS⟩
    /-
      case intro.intro
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.Limits.HasPullbacks C
      K : CategoryTheory.Pretopology C
      X Y : C
      S : CategoryTheory.Sieve X
      g : Quiver.Hom Y X
      R : CategoryTheory.Presieve X
      hR : Membership.mem (K.coverings X) R
      RS : LE.le R S.arrows
      ⊢ Membership.mem ((fun X S => Exists fun R => And (Membership.mem (K.coverings …
    -/
    refine ⟨_, K.pullbacks g _ hR, ?_⟩
    /-
      case intro.intro
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.Limits.HasPullbacks C
      K : CategoryTheory.Pretopology C
      X Y : C
      S : CategoryTheory.Sieve X
      g : Quiver.Hom Y X
      R : CategoryTheory.Presieve X
      hR : Membership.mem (K.coverings X) R
      RS : LE.le R S.arrows
      ⊢ LE.le (CategoryTheory.Presieve.pullbackArrows g R) (CategoryTheory.Sieve.pul …
    -/
    rw [← Sieve.generate_le_iff, Sieve.pullbackArrows_comm]
    /-
      case intro.intro
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.Limits.HasPullbacks C
      K : CategoryTheory.Pretopology C
      X Y : C
      S : CategoryTheory.Sieve X
      g : Quiver.Hom Y X
      R : CategoryTheory.Presieve X
      hR : Membership.mem (K.coverings X) R
      RS : LE.le R S.arrows
      ⊢ LE.le (CategoryTheory.Sieve.pullback g (CategoryTheory.Sieve.generate R)) (C …
    -/
    apply Sieve.pullback_monotone
    /-
      case intro.intro.a
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.Limits.HasPullbacks C
      K : CategoryTheory.Pretopology C
      X Y : C
      S : CategoryTheory.Sieve X
      g : Quiver.Hom Y X
      R : CategoryTheory.Presieve X
      hR : Membership.mem (K.coverings X) R
      RS : LE.le R S.arrows
      ⊢ LE.le (CategoryTheory.Sieve.generate R) S
    -/
    rwa [Sieve.giGenerate.gc]
    /-
      🎉 no goals
    -/
  transitive' := by
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.Limits.HasPullbacks C
      K : CategoryTheory.Pretopology C
      ⊢ ∀ ⦃X : C⦄ ⦃S : CategoryTheory.Sieve X⦄, Membership.mem ((fun X S => Exists f …
    -/
    rintro X S ⟨R', hR', RS⟩ R t
    /-
      case intro.intro
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.Limits.HasPullbacks C
      K : CategoryTheory.Pretopology C
      X : C
      S : CategoryTheory.Sieve X
      R' : CategoryTheory.Presieve X
      hR' : Membership.mem (K.coverings X) R'
      RS : LE.le R' S.arrows
      R : CategoryTheory.Sieve X
      t : ∀ ⦃Y : C⦄ ⦃f : Quiver.Hom Y X⦄, S.arrows f → Membership.mem ((fun X S => E …
      ⊢ Membership.mem ((fun X S => Exists fun R => And (Membership.mem (K.coverings …
    -/
    choose t₁ t₂ t₃ using t
    /-
      case intro.intro
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.Limits.HasPullbacks C
      K : CategoryTheory.Pretopology C
      X : C
      S : CategoryTheory.Sieve X
      R' : CategoryTheory.Presieve X
      hR' : Membership.mem (K.coverings X) R'
      RS : LE.le R' S.arrows
      R : CategoryTheory.Sieve X
      t₁ : ⦃Y : C⦄ → ⦃f : Quiver.Hom Y X⦄ → S.arrows f → CategoryTheory.Presieve Y
      t₂ : ∀ ⦃Y : C⦄ ⦃f : Quiver.Hom Y X⦄ (a : S.arrows f), Membership.mem (K.coveri …
      t₃ : ∀ ⦃Y : C⦄ ⦃f : Quiver.Hom Y X⦄ (a : S.arrows f), LE.le (t₁ a) (CategoryTh …
      ⊢ Membership.mem ((fun X S => Exists fun R => And (Membership.mem (K.coverings …
    -/
    refine ⟨_, K.transitive _ _ hR' fun _ f hf => t₂ (RS _ hf), ?_⟩
    /-
      case intro.intro
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.Limits.HasPullbacks C
      K : CategoryTheory.Pretopology C
      X : C
      S : CategoryTheory.Sieve X
      R' : CategoryTheory.Presieve X
      hR' : Membership.mem (K.coverings X) R'
      RS : LE.le R' S.arrows
      R : CategoryTheory.Sieve X
      t₁ : ⦃Y : C⦄ → ⦃f : Quiver.Hom Y X⦄ → S.arrows f → CategoryTheory.Presieve Y
      t₂ : ∀ ⦃Y : C⦄ ⦃f : Quiver.Hom Y X⦄ (a : S.arrows f), Membership.mem (K.coveri …
      t₃ : ∀ ⦃Y : C⦄ ⦃f : Quiver.Hom Y X⦄ (a : S.arrows f), LE.le (t₁ a) (CategoryTh …
      ⊢ LE.le (R'.bind fun x f hf => t₁ ⋯) R.arrows
    -/
    rintro Y _ ⟨Z, g, f, hg, hf, rfl⟩
    /-
      case intro.intro.intro.intro.intro.intro.intro
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.Limits.HasPullbacks C
      K : CategoryTheory.Pretopology C
      X : C
      S : CategoryTheory.Sieve X
      R' : CategoryTheory.Presieve X
      hR' : Membership.mem (K.coverings X) R'
      RS : LE.le R' S.arrows
      R : CategoryTheory.Sieve X
      t₁ : ⦃Y : C⦄ → ⦃f : Quiver.Hom Y X⦄ → S.arrows f → CategoryTheory.Presieve Y
      t₂ : ∀ ⦃Y : C⦄ ⦃f : Quiver.Hom Y X⦄ (a : S.arrows f), Membership.mem (K.coveri …
      t₃ : ∀ ⦃Y : C⦄ ⦃f : Quiver.Hom Y X⦄ (a : S.arrows f), LE.le (t₁ a) (CategoryTh …
      Y Z : C
      g : Quiver.Hom Y Z
      f : Quiver.Hom Z X
      hg : R' f
      hf : t₁ ⋯ g
      ⊢ Membership.mem R.arrows (CategoryTheory.CategoryStruct.comp g f)
    -/
    apply t₃ (RS _ hg) _ hf
    /-
      🎉 no goals
    -/


theorem mem_toGrothendieck (K : Pretopology C) (X S) :
    S ∈ toGrothendieck C K X ↔ ∃ R ∈ K X, R ≤ (S : Presieve X) :=
  Iff.rfl


/-- The largest pretopology generating the given Grothendieck topology.

See [MM92] Chapter III, Section 2, Equations (3,4).
-/
def ofGrothendieck (J : GrothendieckTopology C) : Pretopology C where
  coverings X R := Sieve.generate R ∈ J X
                                               /-
                                                 C : Type u
                                                 inst✝¹ : CategoryTheory.Category.{v, u} C
                                                 inst✝ : CategoryTheory.Limits.HasPullbacks C
                                                 J : CategoryTheory.GrothendieckTopology C
                                                 X Y : C
                                                 f : Quiver.Hom Y X
                                                 i : CategoryTheory.IsIso f
                                                 ⊢ Eq (CategoryTheory.Sieve.generate (CategoryTheory.Presieve.singleton f)) Top …
                                               -/
  has_isos X Y f i := J.covering_of_eq_top (by simp)
                                               /-
                                                 🎉 no goals
                                               -/
  pullbacks X Y f R hR := by
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.Limits.HasPullbacks C
      J : CategoryTheory.GrothendieckTopology C
      X Y : C
      f : Quiver.Hom Y X
      R : CategoryTheory.Presieve X
      hR : Membership.mem ((fun X R => Membership.mem (J X) (CategoryTheory.Sieve.ge …
      ⊢ Membership.mem ((fun X R => Membership.mem (J X) (CategoryTheory.Sieve.gener …
    -/
    simp only [Set.mem_def, Sieve.pullbackArrows_comm]
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.Limits.HasPullbacks C
      J : CategoryTheory.GrothendieckTopology C
      X Y : C
      f : Quiver.Hom Y X
      R : CategoryTheory.Presieve X
      hR : Membership.mem ((fun X R => Membership.mem (J X) (CategoryTheory.Sieve.ge …
      ⊢ J Y (CategoryTheory.Sieve.pullback f (CategoryTheory.Sieve.generate R))
    -/
    apply J.pullback_stable f hR
    /-
      🎉 no goals
    -/
  transitive X S Ti hS hTi := by
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.Limits.HasPullbacks C
      J : CategoryTheory.GrothendieckTopology C
      X : C
      S : CategoryTheory.Presieve X
      Ti : ⦃Y : C⦄ → (f : Quiver.Hom Y X) → S f → CategoryTheory.Presieve Y
      hS : Membership.mem ((fun X R => Membership.mem (J X) (CategoryTheory.Sieve.ge …
      hTi : ∀ ⦃Y : C⦄ (f : Quiver.Hom Y X) (H : S f), Membership.mem ((fun X R => Me …
      ⊢ Membership.mem ((fun X R => Membership.mem (J X) (CategoryTheory.Sieve.gener …
    -/
    apply J.transitive hS
    /-
      case h
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.Limits.HasPullbacks C
      J : CategoryTheory.GrothendieckTopology C
      X : C
      S : CategoryTheory.Presieve X
      Ti : ⦃Y : C⦄ → (f : Quiver.Hom Y X) → S f → CategoryTheory.Presieve Y
      hS : Membership.mem ((fun X R => Membership.mem (J X) (CategoryTheory.Sieve.ge …
      hTi : ∀ ⦃Y : C⦄ (f : Quiver.Hom Y X) (H : S f), Membership.mem ((fun X R => Me …
      ⊢ ∀ ⦃Y : C⦄ ⦃f : Quiver.Hom Y X⦄, (CategoryTheory.Sieve.generate S).arrows f → …
    -/
    intro Y f
    /-
      case h
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.Limits.HasPullbacks C
      J : CategoryTheory.GrothendieckTopology C
      X : C
      S : CategoryTheory.Presieve X
      Ti : ⦃Y : C⦄ → (f : Quiver.Hom Y X) → S f → CategoryTheory.Presieve Y
      hS : Membership.mem ((fun X R => Membership.mem (J X) (CategoryTheory.Sieve.ge …
      hTi : ∀ ⦃Y : C⦄ (f : Quiver.Hom Y X) (H : S f), Membership.mem ((fun X R => Me …
      Y : C
      f : Quiver.Hom Y X
      ⊢ (CategoryTheory.Sieve.generate S).arrows f → Membership.mem (J Y) (CategoryT …
    -/
    rintro ⟨Z, g, f, hf, rfl⟩
    /-
      case h.intro.intro.intro.intro
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.Limits.HasPullbacks C
      J : CategoryTheory.GrothendieckTopology C
      X : C
      S : CategoryTheory.Presieve X
      Ti : ⦃Y : C⦄ → (f : Quiver.Hom Y X) → S f → CategoryTheory.Presieve Y
      hS : Membership.mem ((fun X R => Membership.mem (J X) (CategoryTheory.Sieve.ge …
      hTi : ∀ ⦃Y : C⦄ (f : Quiver.Hom Y X) (H : S f), Membership.mem ((fun X R => Me …
      Y Z : C
      g : Quiver.Hom Y Z
      f : Quiver.Hom Z X
      hf : S f
      ⊢ Membership.mem (J Y) (CategoryTheory.Sieve.pullback (CategoryTheory.Category …
    -/
    rw [Sieve.pullback_comp]
    /-
      case h.intro.intro.intro.intro
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.Limits.HasPullbacks C
      J : CategoryTheory.GrothendieckTopology C
      X : C
      S : CategoryTheory.Presieve X
      Ti : ⦃Y : C⦄ → (f : Quiver.Hom Y X) → S f → CategoryTheory.Presieve Y
      hS : Membership.mem ((fun X R => Membership.mem (J X) (CategoryTheory.Sieve.ge …
      hTi : ∀ ⦃Y : C⦄ (f : Quiver.Hom Y X) (H : S f), Membership.mem ((fun X R => Me …
      Y Z : C
      g : Quiver.Hom Y Z
      f : Quiver.Hom Z X
      hf : S f
      ⊢ Membership.mem (J Y) (CategoryTheory.Sieve.pullback g (CategoryTheory.Sieve. …
    -/
    apply J.pullback_stable g
    /-
      case h.intro.intro.intro.intro
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.Limits.HasPullbacks C
      J : CategoryTheory.GrothendieckTopology C
      X : C
      S : CategoryTheory.Presieve X
      Ti : ⦃Y : C⦄ → (f : Quiver.Hom Y X) → S f → CategoryTheory.Presieve Y
      hS : Membership.mem ((fun X R => Membership.mem (J X) (CategoryTheory.Sieve.ge …
      hTi : ∀ ⦃Y : C⦄ (f : Quiver.Hom Y X) (H : S f), Membership.mem ((fun X R => Me …
      Y Z : C
      g : Quiver.Hom Y Z
      f : Quiver.Hom Z X
      hf : S f
      ⊢ Membership.mem (J Z) (CategoryTheory.Sieve.pullback f (CategoryTheory.Sieve. …
    -/
    apply J.superset_covering _ (hTi _ hf)
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.Limits.HasPullbacks C
      J : CategoryTheory.GrothendieckTopology C
      X : C
      S : CategoryTheory.Presieve X
      Ti : ⦃Y : C⦄ → (f : Quiver.Hom Y X) → S f → CategoryTheory.Presieve Y
      hS : Membership.mem ((fun X R => Membership.mem (J X) (CategoryTheory.Sieve.ge …
      hTi : ∀ ⦃Y : C⦄ (f : Quiver.Hom Y X) (H : S f), Membership.mem ((fun X R => Me …
      Y Z : C
      g : Quiver.Hom Y Z
      f : Quiver.Hom Z X
      hf : S f
      ⊢ LE.le (CategoryTheory.Sieve.generate (Ti f hf)) (CategoryTheory.Sieve.pullba …
    -/
    rintro Y g ⟨W, h, g, hg, rfl⟩
    /-
      case intro.intro.intro.intro
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.Limits.HasPullbacks C
      J : CategoryTheory.GrothendieckTopology C
      X : C
      S : CategoryTheory.Presieve X
      Ti : ⦃Y : C⦄ → (f : Quiver.Hom Y X) → S f → CategoryTheory.Presieve Y
      hS : Membership.mem ((fun X R => Membership.mem (J X) (CategoryTheory.Sieve.ge …
      hTi : ∀ ⦃Y : C⦄ (f : Quiver.Hom Y X) (H : S f), Membership.mem ((fun X R => Me …
      Y✝ Z : C
      g✝ : Quiver.Hom Y✝ Z
      f : Quiver.Hom Z X
      hf : S f
      Y W : C
      h : Quiver.Hom Y W
      g : Quiver.Hom W Z
      hg : Ti f hf g
      ⊢ (CategoryTheory.Sieve.pullback f (CategoryTheory.Sieve.generate (S.bind Ti)) …
    -/
    exact ⟨_, h, _, ⟨_, _, _, hf, hg, rfl⟩, by simp⟩
    /-
      🎉 no goals
    -/


/-- We have a galois insertion from pretopologies to Grothendieck topologies. -/
def gi : GaloisInsertion (toGrothendieck C) (ofGrothendieck C) where
  gc K J := by
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.Limits.HasPullbacks C
      K : CategoryTheory.Pretopology C
      J : CategoryTheory.GrothendieckTopology C
      ⊢ Iff (LE.le (CategoryTheory.Pretopology.toGrothendieck C K) J) (LE.le K (Cate …
    -/
    constructor
      /-
        case mp
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        inst✝ : CategoryTheory.Limits.HasPullbacks C
        K : CategoryTheory.Pretopology C
        J : CategoryTheory.GrothendieckTopology C
        ⊢ LE.le (CategoryTheory.Pretopology.toGrothendieck C K) J → LE.le K (CategoryT …
      -/
    · intro h X R hR
      /-
        case mp
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        inst✝ : CategoryTheory.Limits.HasPullbacks C
        K : CategoryTheory.Pretopology C
        J : CategoryTheory.GrothendieckTopology C
        h : LE.le (CategoryTheory.Pretopology.toGrothendieck C K) J
        X : C
        R : CategoryTheory.Presieve X
        hR : Membership.mem (K.coverings X) R
        ⊢ Membership.mem ((CategoryTheory.Pretopology.ofGrothendieck C J).coverings X) R
      -/
      exact h _ ⟨_, hR, Sieve.le_generate R⟩
      /-
        🎉 no goals
      -/
      /-
        case mpr
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        inst✝ : CategoryTheory.Limits.HasPullbacks C
        K : CategoryTheory.Pretopology C
        J : CategoryTheory.GrothendieckTopology C
        ⊢ LE.le K (CategoryTheory.Pretopology.ofGrothendieck C J) → LE.le (CategoryThe …
      -/
    · rintro h X S ⟨R, hR, RS⟩
      /-
        case mpr.intro.intro
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        inst✝ : CategoryTheory.Limits.HasPullbacks C
        K : CategoryTheory.Pretopology C
        J : CategoryTheory.GrothendieckTopology C
        h : LE.le K (CategoryTheory.Pretopology.ofGrothendieck C J)
        X : C
        S : CategoryTheory.Sieve X
        R : CategoryTheory.Presieve X
        hR : Membership.mem (K.coverings X) R
        RS : LE.le R S.arrows
        ⊢ Membership.mem (J X) S
      -/
      apply J.superset_covering _ (h _ hR)
      /-
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        inst✝ : CategoryTheory.Limits.HasPullbacks C
        K : CategoryTheory.Pretopology C
        J : CategoryTheory.GrothendieckTopology C
        h : LE.le K (CategoryTheory.Pretopology.ofGrothendieck C J)
        X : C
        S : CategoryTheory.Sieve X
        R : CategoryTheory.Presieve X
        hR : Membership.mem (K.coverings X) R
        RS : LE.le R S.arrows
        ⊢ LE.le (CategoryTheory.Sieve.generate R) S
      -/
      rwa [Sieve.giGenerate.gc]
      /-
        🎉 no goals
      -/
  le_l_u J _ S hS := ⟨S, J.superset_covering (Sieve.le_generate S.arrows) hS, le_rfl⟩
  choice x _ := toGrothendieck C x
  choice_eq _ _ := rfl


lemma mem_ofGrothendieck (t : GrothendieckTopology C) {X : C} (S : Presieve X) :
    S ∈ ofGrothendieck C t X ↔ Sieve.generate S ∈ t X :=
  Iff.rfl


/--
The trivial pretopology, in which the coverings are exactly singleton isomorphisms. This topology is
also known as the indiscrete, coarse, or chaotic topology.

See <https://stacks.math.columbia.edu/tag/07GE>
-/
def trivial : Pretopology C where
  coverings X S := ∃ (Y : _) (f : Y ⟶ X) (_ : IsIso f), S = Presieve.singleton f
  has_isos _ _ _ i := ⟨_, _, i, rfl⟩
  pullbacks X Y f S := by
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.Limits.HasPullbacks C
      X Y : C
      f : Quiver.Hom Y X
      S : CategoryTheory.Presieve X
      ⊢ Membership.mem ((fun X S => Exists fun Y => Exists fun f => Exists fun x =>  …
    -/
    rintro ⟨Z, g, i, rfl⟩
    /-
      case intro.intro.intro
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.Limits.HasPullbacks C
      X Y : C
      f : Quiver.Hom Y X
      Z : C
      g : Quiver.Hom Z X
      i : CategoryTheory.IsIso g
      ⊢ Membership.mem ((fun X S => Exists fun Y => Exists fun f => Exists fun x =>  …
    -/
    refine ⟨pullback g f, pullback.snd _ _, ?_, ?_⟩
      /-
        case intro.intro.intro.refine_1
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        inst✝ : CategoryTheory.Limits.HasPullbacks C
        X Y : C
        f : Quiver.Hom Y X
        Z : C
        g : Quiver.Hom Z X
        i : CategoryTheory.IsIso g
        ⊢ CategoryTheory.IsIso (CategoryTheory.Limits.pullback.snd g f)
      -/
    · refine ⟨⟨pullback.lift (f ≫ inv g) (𝟙 _) (by simp), ⟨?_, by aesop_cat⟩⟩⟩
      /-
        case intro.intro.intro.refine_1
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        inst✝ : CategoryTheory.Limits.HasPullbacks C
        X Y : C
        f : Quiver.Hom Y X
        Z : C
        g : Quiver.Hom Z X
        i : CategoryTheory.IsIso g
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullback.snd g …
      -/
      ext
        /-
          case intro.intro.intro.refine_1.h₀
          C : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} C
          inst✝ : CategoryTheory.Limits.HasPullbacks C
          X Y : C
          f : Quiver.Hom Y X
          Z : C
          g : Quiver.Hom Z X
          i : CategoryTheory.IsIso g
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
        -/
      · rw [assoc, pullback.lift_fst, ← pullback.condition_assoc]
        /-
          case intro.intro.intro.refine_1.h₀
          C : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} C
          inst✝ : CategoryTheory.Limits.HasPullbacks C
          X Y : C
          f : Quiver.Hom Y X
          Z : C
          g : Quiver.Hom Z X
          i : CategoryTheory.IsIso g
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullback.fst g …
        -/
        simp
        /-
          🎉 no goals
        -/
        /-
          case intro.intro.intro.refine_1.h₁
          C : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} C
          inst✝ : CategoryTheory.Limits.HasPullbacks C
          X Y : C
          f : Quiver.Hom Y X
          Z : C
          g : Quiver.Hom Z X
          i : CategoryTheory.IsIso g
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
        -/
      · simp
        /-
          🎉 no goals
        -/
      /-
        case intro.intro.intro.refine_2
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        inst✝ : CategoryTheory.Limits.HasPullbacks C
        X Y : C
        f : Quiver.Hom Y X
        Z : C
        g : Quiver.Hom Z X
        i : CategoryTheory.IsIso g
        ⊢ Eq (CategoryTheory.Presieve.pullbackArrows f (CategoryTheory.Presieve.single …
      -/
    · apply pullback_singleton
      /-
        🎉 no goals
      -/
  transitive := by
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.Limits.HasPullbacks C
      ⊢ ∀ ⦃X : C⦄ (S : CategoryTheory.Presieve X) (Ti : ⦃Y : C⦄ → (f : Quiver.Hom Y  …
    -/
    rintro X S Ti ⟨Z, g, i, rfl⟩ hS
    /-
      case intro.intro.intro
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.Limits.HasPullbacks C
      X Z : C
      g : Quiver.Hom Z X
      i : CategoryTheory.IsIso g
      Ti : ⦃Y : C⦄ → (f : Quiver.Hom Y X) → CategoryTheory.Presieve.singleton g f →  …
      hS : ∀ ⦃Y : C⦄ (f : Quiver.Hom Y X) (H : CategoryTheory.Presieve.singleton g f …
      ⊢ Membership.mem ((fun X S => Exists fun Y => Exists fun f => Exists fun x =>  …
    -/
    rcases hS g (singleton_self g) with ⟨Y, f, i, hTi⟩
    /-
      case intro.intro.intro.intro.intro.intro
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.Limits.HasPullbacks C
      X Z : C
      g : Quiver.Hom Z X
      i✝ : CategoryTheory.IsIso g
      Ti : ⦃Y : C⦄ → (f : Quiver.Hom Y X) → CategoryTheory.Presieve.singleton g f →  …
      hS : ∀ ⦃Y : C⦄ (f : Quiver.Hom Y X) (H : CategoryTheory.Presieve.singleton g f …
      Y : C
      f : Quiver.Hom Y Z
      i : CategoryTheory.IsIso f
      hTi : Eq (Ti g ⋯) (CategoryTheory.Presieve.singleton f)
      ⊢ Membership.mem ((fun X S => Exists fun Y => Exists fun f => Exists fun x =>  …
    -/
    refine ⟨_, f ≫ g, ?_, ?_⟩
      /-
        case intro.intro.intro.intro.intro.intro.refine_1
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        inst✝ : CategoryTheory.Limits.HasPullbacks C
        X Z : C
        g : Quiver.Hom Z X
        i✝ : CategoryTheory.IsIso g
        Ti : ⦃Y : C⦄ → (f : Quiver.Hom Y X) → CategoryTheory.Presieve.singleton g f →  …
        hS : ∀ ⦃Y : C⦄ (f : Quiver.Hom Y X) (H : CategoryTheory.Presieve.singleton g f …
        Y : C
        f : Quiver.Hom Y Z
        i : CategoryTheory.IsIso f
        hTi : Eq (Ti g ⋯) (CategoryTheory.Presieve.singleton f)
        ⊢ CategoryTheory.IsIso (CategoryTheory.CategoryStruct.comp f g)
      -/
    · infer_instance
      /-
        🎉 no goals
      -/
    -- Porting note (https://github.com/leanprover-community/mathlib4/issues/11041): the next four lines were just "ext (W k)"
    /-
      case intro.intro.intro.intro.intro.intro.refine_2
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.Limits.HasPullbacks C
      X Z : C
      g : Quiver.Hom Z X
      i✝ : CategoryTheory.IsIso g
      Ti : ⦃Y : C⦄ → (f : Quiver.Hom Y X) → CategoryTheory.Presieve.singleton g f →  …
      hS : ∀ ⦃Y : C⦄ (f : Quiver.Hom Y X) (H : CategoryTheory.Presieve.singleton g f …
      Y : C
      f : Quiver.Hom Y Z
      i : CategoryTheory.IsIso f
      hTi : Eq (Ti g ⋯) (CategoryTheory.Presieve.singleton f)
      ⊢ Eq ((CategoryTheory.Presieve.singleton g).bind Ti) (CategoryTheory.Presieve. …
    -/
    apply funext
    /-
      case intro.intro.intro.intro.intro.intro.refine_2.h
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.Limits.HasPullbacks C
      X Z : C
      g : Quiver.Hom Z X
      i✝ : CategoryTheory.IsIso g
      Ti : ⦃Y : C⦄ → (f : Quiver.Hom Y X) → CategoryTheory.Presieve.singleton g f →  …
      hS : ∀ ⦃Y : C⦄ (f : Quiver.Hom Y X) (H : CategoryTheory.Presieve.singleton g f …
      Y : C
      f : Quiver.Hom Y Z
      i : CategoryTheory.IsIso f
      hTi : Eq (Ti g ⋯) (CategoryTheory.Presieve.singleton f)
      ⊢ ∀ (x : C), Eq ((CategoryTheory.Presieve.singleton g).bind Ti) (CategoryTheor …
    -/
    rintro W
    /-
      case intro.intro.intro.intro.intro.intro.refine_2.h
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.Limits.HasPullbacks C
      X Z : C
      g : Quiver.Hom Z X
      i✝ : CategoryTheory.IsIso g
      Ti : ⦃Y : C⦄ → (f : Quiver.Hom Y X) → CategoryTheory.Presieve.singleton g f →  …
      hS : ∀ ⦃Y : C⦄ (f : Quiver.Hom Y X) (H : CategoryTheory.Presieve.singleton g f …
      Y : C
      f : Quiver.Hom Y Z
      i : CategoryTheory.IsIso f
      hTi : Eq (Ti g ⋯) (CategoryTheory.Presieve.singleton f)
      W : C
      ⊢ Eq ((CategoryTheory.Presieve.singleton g).bind Ti) (CategoryTheory.Presieve. …
    -/
    apply Set.ext
    /-
      case intro.intro.intro.intro.intro.intro.refine_2.h.h
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.Limits.HasPullbacks C
      X Z : C
      g : Quiver.Hom Z X
      i✝ : CategoryTheory.IsIso g
      Ti : ⦃Y : C⦄ → (f : Quiver.Hom Y X) → CategoryTheory.Presieve.singleton g f →  …
      hS : ∀ ⦃Y : C⦄ (f : Quiver.Hom Y X) (H : CategoryTheory.Presieve.singleton g f …
      Y : C
      f : Quiver.Hom Y Z
      i : CategoryTheory.IsIso f
      hTi : Eq (Ti g ⋯) (CategoryTheory.Presieve.singleton f)
      W : C
      ⊢ ∀ (x : Quiver.Hom W X), Iff (Membership.mem ((CategoryTheory.Presieve.single …
    -/
    rintro k
    /-
      case intro.intro.intro.intro.intro.intro.refine_2.h.h
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.Limits.HasPullbacks C
      X Z : C
      g : Quiver.Hom Z X
      i✝ : CategoryTheory.IsIso g
      Ti : ⦃Y : C⦄ → (f : Quiver.Hom Y X) → CategoryTheory.Presieve.singleton g f →  …
      hS : ∀ ⦃Y : C⦄ (f : Quiver.Hom Y X) (H : CategoryTheory.Presieve.singleton g f …
      Y : C
      f : Quiver.Hom Y Z
      i : CategoryTheory.IsIso f
      hTi : Eq (Ti g ⋯) (CategoryTheory.Presieve.singleton f)
      W : C
      k : Quiver.Hom W X
      ⊢ Iff (Membership.mem ((CategoryTheory.Presieve.singleton g).bind Ti) k) (Memb …
    -/
    constructor
      /-
        case intro.intro.intro.intro.intro.intro.refine_2.h.h.mp
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        inst✝ : CategoryTheory.Limits.HasPullbacks C
        X Z : C
        g : Quiver.Hom Z X
        i✝ : CategoryTheory.IsIso g
        Ti : ⦃Y : C⦄ → (f : Quiver.Hom Y X) → CategoryTheory.Presieve.singleton g f →  …
        hS : ∀ ⦃Y : C⦄ (f : Quiver.Hom Y X) (H : CategoryTheory.Presieve.singleton g f …
        Y : C
        f : Quiver.Hom Y Z
        i : CategoryTheory.IsIso f
        hTi : Eq (Ti g ⋯) (CategoryTheory.Presieve.singleton f)
        W : C
        k : Quiver.Hom W X
        ⊢ Membership.mem ((CategoryTheory.Presieve.singleton g).bind Ti) k → Membershi …
      -/
    · rintro ⟨V, h, k, ⟨_⟩, hh, rfl⟩
      /-
        case intro.intro.intro.intro.intro.intro.refine_2.h.h.mp.intro.intro.intro.int …
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        inst✝ : CategoryTheory.Limits.HasPullbacks C
        X Z : C
        g : Quiver.Hom Z X
        i✝ : CategoryTheory.IsIso g
        Ti : ⦃Y : C⦄ → (f : Quiver.Hom Y X) → CategoryTheory.Presieve.singleton g f →  …
        hS : ∀ ⦃Y : C⦄ (f : Quiver.Hom Y X) (H : CategoryTheory.Presieve.singleton g f …
        Y✝ : C
        f : Quiver.Hom Y✝ Z
        i : CategoryTheory.IsIso f
        hTi : Eq (Ti g ⋯) (CategoryTheory.Presieve.singleton f)
        W Y : C
        h : Quiver.Hom W Z
        hh : Ti g ⋯ h
        ⊢ Membership.mem (CategoryTheory.Presieve.singleton (CategoryTheory.CategorySt …
      -/
      rw [hTi] at hh
      /-
        case intro.intro.intro.intro.intro.intro.refine_2.h.h.mp.intro.intro.intro.int …
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        inst✝ : CategoryTheory.Limits.HasPullbacks C
        X Z : C
        g : Quiver.Hom Z X
        i✝ : CategoryTheory.IsIso g
        Ti : ⦃Y : C⦄ → (f : Quiver.Hom Y X) → CategoryTheory.Presieve.singleton g f →  …
        hS : ∀ ⦃Y : C⦄ (f : Quiver.Hom Y X) (H : CategoryTheory.Presieve.singleton g f …
        Y✝ : C
        f : Quiver.Hom Y✝ Z
        i : CategoryTheory.IsIso f
        hTi : Eq (Ti g ⋯) (CategoryTheory.Presieve.singleton f)
        W Y : C
        h : Quiver.Hom W Z
        hh : CategoryTheory.Presieve.singleton f h
        ⊢ Membership.mem (CategoryTheory.Presieve.singleton (CategoryTheory.CategorySt …
      -/
      cases hh
      /-
        case intro.intro.intro.intro.intro.intro.refine_2.h.h.mp.intro.intro.intro.int …
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        inst✝ : CategoryTheory.Limits.HasPullbacks C
        X Z : C
        g : Quiver.Hom Z X
        i✝ : CategoryTheory.IsIso g
        Ti : ⦃Y : C⦄ → (f : Quiver.Hom Y X) → CategoryTheory.Presieve.singleton g f →  …
        hS : ∀ ⦃Y : C⦄ (f : Quiver.Hom Y X) (H : CategoryTheory.Presieve.singleton g f …
        Y✝ : C
        f : Quiver.Hom Y✝ Z
        i : CategoryTheory.IsIso f
        hTi : Eq (Ti g ⋯) (CategoryTheory.Presieve.singleton f)
        Y : C
        ⊢ Membership.mem (CategoryTheory.Presieve.singleton (CategoryTheory.CategorySt …
      -/
      apply singleton.mk
      /-
        🎉 no goals
      -/
      /-
        case intro.intro.intro.intro.intro.intro.refine_2.h.h.mpr
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        inst✝ : CategoryTheory.Limits.HasPullbacks C
        X Z : C
        g : Quiver.Hom Z X
        i✝ : CategoryTheory.IsIso g
        Ti : ⦃Y : C⦄ → (f : Quiver.Hom Y X) → CategoryTheory.Presieve.singleton g f →  …
        hS : ∀ ⦃Y : C⦄ (f : Quiver.Hom Y X) (H : CategoryTheory.Presieve.singleton g f …
        Y : C
        f : Quiver.Hom Y Z
        i : CategoryTheory.IsIso f
        hTi : Eq (Ti g ⋯) (CategoryTheory.Presieve.singleton f)
        W : C
        k : Quiver.Hom W X
        ⊢ Membership.mem (CategoryTheory.Presieve.singleton (CategoryTheory.CategorySt …
      -/
    · rintro ⟨_⟩
      /-
        case intro.intro.intro.intro.intro.intro.refine_2.h.h.mpr.mk
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        inst✝ : CategoryTheory.Limits.HasPullbacks C
        X Z : C
        g : Quiver.Hom Z X
        i✝ : CategoryTheory.IsIso g
        Ti : ⦃Y : C⦄ → (f : Quiver.Hom Y X) → CategoryTheory.Presieve.singleton g f →  …
        hS : ∀ ⦃Y : C⦄ (f : Quiver.Hom Y X) (H : CategoryTheory.Presieve.singleton g f …
        Y✝ : C
        f : Quiver.Hom Y✝ Z
        i : CategoryTheory.IsIso f
        hTi : Eq (Ti g ⋯) (CategoryTheory.Presieve.singleton f)
        Y : C
        ⊢ Membership.mem ((CategoryTheory.Presieve.singleton g).bind Ti) (CategoryTheo …
      -/
      refine bind_comp g singleton.mk ?_
      /-
        case intro.intro.intro.intro.intro.intro.refine_2.h.h.mpr.mk
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        inst✝ : CategoryTheory.Limits.HasPullbacks C
        X Z : C
        g : Quiver.Hom Z X
        i✝ : CategoryTheory.IsIso g
        Ti : ⦃Y : C⦄ → (f : Quiver.Hom Y X) → CategoryTheory.Presieve.singleton g f →  …
        hS : ∀ ⦃Y : C⦄ (f : Quiver.Hom Y X) (H : CategoryTheory.Presieve.singleton g f …
        Y✝ : C
        f : Quiver.Hom Y✝ Z
        i : CategoryTheory.IsIso f
        hTi : Eq (Ti g ⋯) (CategoryTheory.Presieve.singleton f)
        Y : C
        ⊢ Ti g ⋯ f
      -/
      rw [hTi]
      /-
        case intro.intro.intro.intro.intro.intro.refine_2.h.h.mpr.mk
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        inst✝ : CategoryTheory.Limits.HasPullbacks C
        X Z : C
        g : Quiver.Hom Z X
        i✝ : CategoryTheory.IsIso g
        Ti : ⦃Y : C⦄ → (f : Quiver.Hom Y X) → CategoryTheory.Presieve.singleton g f →  …
        hS : ∀ ⦃Y : C⦄ (f : Quiver.Hom Y X) (H : CategoryTheory.Presieve.singleton g f …
        Y✝ : C
        f : Quiver.Hom Y✝ Z
        i : CategoryTheory.IsIso f
        hTi : Eq (Ti g ⋯) (CategoryTheory.Presieve.singleton f)
        Y : C
        ⊢ CategoryTheory.Presieve.singleton f f
      -/
      apply singleton.mk
      /-
        🎉 no goals
      -/


instance orderBot : OrderBot (Pretopology C) where
  bot := trivial C
  bot_le K X R := by
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.Limits.HasPullbacks C
      K : CategoryTheory.Pretopology C
      X : C
      R : CategoryTheory.Presieve X
      ⊢ Membership.mem (Bot.bot.coverings X) R → Membership.mem (K.coverings X) R
    -/
    rintro ⟨Y, f, hf, rfl⟩
    /-
      case intro.intro.intro
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.Limits.HasPullbacks C
      K : CategoryTheory.Pretopology C
      X Y : C
      f : Quiver.Hom Y X
      hf : CategoryTheory.IsIso f
      ⊢ Membership.mem (K.coverings X) (CategoryTheory.Presieve.singleton f)
    -/
    exact K.has_isos f
    /-
      🎉 no goals
    -/


/-- The trivial pretopology induces the trivial grothendieck topology. -/
theorem toGrothendieck_bot : toGrothendieck C ⊥ = ⊥ :=
  (gi C).gc.l_bot


instance : InfSet (Pretopology C) where
  sInf T := {
    coverings := sInf (coverings '' T)
    has_isos := fun X Y f _ ↦ by
      simp only [sInf_apply, Set.iInf_eq_iInter, Set.iInter_coe_set, Set.mem_image,
        Set.iInter_exists,
        Set.biInter_and', Set.iInter_iInter_eq_right, Set.mem_iInter]
      /-
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        inst✝ : CategoryTheory.Limits.HasPullbacks C
        T : Set (CategoryTheory.Pretopology C)
        X Y : C
        f : Quiver.Hom Y X
        x✝ : CategoryTheory.IsIso f
        ⊢ ∀ (i : CategoryTheory.Pretopology C), Membership.mem T i → Membership.mem (i …
      -/
      intro t _
      /-
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        inst✝ : CategoryTheory.Limits.HasPullbacks C
        T : Set (CategoryTheory.Pretopology C)
        X Y : C
        f : Quiver.Hom Y X
        x✝ : CategoryTheory.IsIso f
        t : CategoryTheory.Pretopology C
        i✝ : Membership.mem T t
        ⊢ Membership.mem (t.coverings X) (CategoryTheory.Presieve.singleton f)
      -/
      exact t.has_isos f
      /-
        🎉 no goals
      -/
    pullbacks := fun X Y f S hS ↦ by
      simp only [sInf_apply, Set.iInf_eq_iInter, Set.iInter_coe_set, Set.mem_image,
        Set.iInter_exists, Set.biInter_and', Set.iInter_iInter_eq_right, Set.mem_iInter] at hS ⊢
      /-
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        inst✝ : CategoryTheory.Limits.HasPullbacks C
        T : Set (CategoryTheory.Pretopology C)
        X Y : C
        f : Quiver.Hom Y X
        S : CategoryTheory.Presieve X
        hS : ∀ (i : CategoryTheory.Pretopology C), Membership.mem T i → Membership.mem …
        ⊢ ∀ (i : CategoryTheory.Pretopology C), Membership.mem T i → Membership.mem (i …
      -/
      intro t ht
      /-
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        inst✝ : CategoryTheory.Limits.HasPullbacks C
        T : Set (CategoryTheory.Pretopology C)
        X Y : C
        f : Quiver.Hom Y X
        S : CategoryTheory.Presieve X
        hS : ∀ (i : CategoryTheory.Pretopology C), Membership.mem T i → Membership.mem …
        t : CategoryTheory.Pretopology C
        ht : Membership.mem T t
        ⊢ Membership.mem (t.coverings Y) (CategoryTheory.Presieve.pullbackArrows f S)
      -/
      exact t.pullbacks f S (hS t ht)
      /-
        🎉 no goals
      -/
    transitive := fun X S Ti hS hTi ↦ by
      simp only [sInf_apply, Set.iInf_eq_iInter, Set.iInter_coe_set, Set.mem_image,
        Set.iInter_exists, Set.biInter_and', Set.iInter_iInter_eq_right, Set.mem_iInter] at hS hTi ⊢
      /-
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        inst✝ : CategoryTheory.Limits.HasPullbacks C
        T : Set (CategoryTheory.Pretopology C)
        X : C
        S : CategoryTheory.Presieve X
        Ti : ⦃Y : C⦄ → (f : Quiver.Hom Y X) → S f → CategoryTheory.Presieve Y
        hS : ∀ (i : CategoryTheory.Pretopology C), Membership.mem T i → Membership.mem …
        hTi : ∀ ⦃Y : C⦄ (f : Quiver.Hom Y X) (H : S f) (i : CategoryTheory.Pretopology …
        ⊢ ∀ (i : CategoryTheory.Pretopology C), Membership.mem T i → Membership.mem (i …
      -/
      intro t ht
      /-
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        inst✝ : CategoryTheory.Limits.HasPullbacks C
        T : Set (CategoryTheory.Pretopology C)
        X : C
        S : CategoryTheory.Presieve X
        Ti : ⦃Y : C⦄ → (f : Quiver.Hom Y X) → S f → CategoryTheory.Presieve Y
        hS : ∀ (i : CategoryTheory.Pretopology C), Membership.mem T i → Membership.mem …
        hTi : ∀ ⦃Y : C⦄ (f : Quiver.Hom Y X) (H : S f) (i : CategoryTheory.Pretopology …
        t : CategoryTheory.Pretopology C
        ht : Membership.mem T t
        ⊢ Membership.mem (t.coverings X) (S.bind Ti)
      -/
      exact t.transitive S Ti (hS t ht) (fun Y f H ↦ hTi f H t ht)
      /-
        🎉 no goals
      -/
  }


lemma mem_sInf (T : Set (Pretopology C)) {X : C} (S : Presieve X) :
    S ∈ sInf T X ↔ ∀ t ∈ T, S ∈ t X := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Limits.HasPullbacks C
    T : Set (CategoryTheory.Pretopology C)
    X : C
    S : CategoryTheory.Presieve X
    ⊢ Iff (Membership.mem ((InfSet.sInf T).coverings X) S) (∀ (t : CategoryTheory. …
  -/
  show S ∈ sInf (Pretopology.coverings '' T) X ↔ _
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Limits.HasPullbacks C
    T : Set (CategoryTheory.Pretopology C)
    X : C
    S : CategoryTheory.Presieve X
    ⊢ Iff (Membership.mem (InfSet.sInf (Set.image CategoryTheory.Pretopology.cover …
  -/
  simp
  /-
    🎉 no goals
  -/


lemma sInf_ofGrothendieck (T : Set (GrothendieckTopology C)) :
    ofGrothendieck C (sInf T) = sInf (ofGrothendieck C '' T) := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Limits.HasPullbacks C
    T : Set (CategoryTheory.GrothendieckTopology C)
    ⊢ Eq (CategoryTheory.Pretopology.ofGrothendieck C (InfSet.sInf T)) (InfSet.sIn …
  -/
  ext X S
  /-
    case coverings.h.h
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Limits.HasPullbacks C
    T : Set (CategoryTheory.GrothendieckTopology C)
    X : C
    S : CategoryTheory.Presieve X
    ⊢ Iff (Membership.mem ((CategoryTheory.Pretopology.ofGrothendieck C (InfSet.sI …
  -/
  simp [mem_sInf, mem_ofGrothendieck, GrothendieckTopology.mem_sInf]
  /-
    🎉 no goals
  -/


lemma isGLB_sInf (T : Set (Pretopology C)) : IsGLB T (sInf T) :=
  IsGLB.of_image (f := coverings) Iff.rfl (_root_.isGLB_sInf _)


/-- The complete lattice structure on pretopologies. This is induced by the `InfSet` instance, but
with good definitional equalities for `⊥`, `⊤` and `⊓`. -/
instance : CompleteLattice (Pretopology C) where
  __ := orderBot C
  __ := orderTop C
  inf t₁ t₂ := {
    coverings := fun X ↦ t₁.coverings X ∩ t₂.coverings X
    has_isos := fun _ _ f _ ↦
      ⟨t₁.has_isos f, t₂.has_isos f⟩
    pullbacks := fun _ _ f S hS ↦
      ⟨t₁.pullbacks f S hS.left, t₂.pullbacks f S hS.right⟩
    transitive := fun _ S Ti hS hTi ↦
      ⟨t₁.transitive S Ti hS.left (fun _ f H ↦ (hTi f H).left),
        t₂.transitive S Ti hS.right (fun _ f H ↦ (hTi f H).right)⟩
  }
  inf_le_left _ _ _ _ hS := hS.left
  inf_le_right _ _ _ _ hS := hS.right
  le_inf _ _ _ hts htr X _ hS := ⟨hts X hS, htr X hS⟩
  __ := completeLatticeOfInf _ (isGLB_sInf C)


lemma mem_inf (t₁ t₂ : Pretopology C) {X : C} (S : Presieve X) :
    S ∈ (t₁ ⊓ t₂) X ↔ S ∈ t₁ X ∧ S ∈ t₂ X :=
  Iff.rfl


