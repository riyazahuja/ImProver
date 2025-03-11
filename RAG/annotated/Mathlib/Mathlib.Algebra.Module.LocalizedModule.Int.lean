/-- Given `x : M'`, `M'` a localization of `M` via `f`, `IsInteger f x` iff `x` is in the image of
the localization map `f`. -/
def IsInteger (x : M') : Prop :=
  x ∈ LinearMap.range f


lemma isInteger_zero : IsInteger f (0 : M') :=
  Submodule.zero_mem _


theorem isInteger_add {x y : M'} (hx : IsInteger f x) (hy : IsInteger f y) : IsInteger f (x + y) :=
  Submodule.add_mem _ hx hy


theorem isInteger_smul {a : R} {x : M'} (hx : IsInteger f x) : IsInteger f (a • x) := by
  /-
    R : Type u_1
    inst✝⁴ : CommSemiring R
    M : Type u_2
    inst✝³ : AddCommMonoid M
    inst✝² : Module R M
    M' : Type u_3
    inst✝¹ : AddCommMonoid M'
    inst✝ : Module R M'
    f : LinearMap (RingHom.id R) M M'
    a : R
    x : M'
    hx : IsLocalizedModule.IsInteger f x
    ⊢ IsLocalizedModule.IsInteger f (HSMul.hSMul a x)
  -/
  rcases hx with ⟨x', hx⟩
  /-
    case intro
    R : Type u_1
    inst✝⁴ : CommSemiring R
    M : Type u_2
    inst✝³ : AddCommMonoid M
    inst✝² : Module R M
    M' : Type u_3
    inst✝¹ : AddCommMonoid M'
    inst✝ : Module R M'
    f : LinearMap (RingHom.id R) M M'
    a : R
    x : M'
    x' : M
    hx : Eq (f x') x
    ⊢ IsLocalizedModule.IsInteger f (HSMul.hSMul a x)
  -/
  use a • x'
  /-
    case h
    R : Type u_1
    inst✝⁴ : CommSemiring R
    M : Type u_2
    inst✝³ : AddCommMonoid M
    inst✝² : Module R M
    M' : Type u_3
    inst✝¹ : AddCommMonoid M'
    inst✝ : Module R M'
    f : LinearMap (RingHom.id R) M M'
    a : R
    x : M'
    x' : M
    hx : Eq (f x') x
    ⊢ Eq (f (HSMul.hSMul a x')) (HSMul.hSMul a x)
  -/
  rw [← hx, LinearMapClass.map_smul]
  /-
    🎉 no goals
  -/


/-- Each element `x : M'` has an `S`-multiple which is an integer. -/
theorem exists_integer_multiple (x : M') : ∃ a : S, IsInteger f (a.val • x) :=
  let ⟨⟨Num, denom⟩, h⟩ := IsLocalizedModule.surj S f x
  ⟨denom, Set.mem_range.mpr ⟨Num, h.symm⟩⟩


/-- We can clear the denominators of a `Finset`-indexed family of fractions. -/
theorem exist_integer_multiples {ι : Type*} (s : Finset ι) (g : ι → M') :
    ∃ b : S, ∀ i ∈ s, IsInteger f (b.val • g i) := by
  classical
  choose sec hsec using (fun i ↦ IsLocalizedModule.surj S f (g i))
  refine ⟨∏ i ∈ s, (sec i).2, fun i hi => ⟨?_, ?_⟩⟩
  · exact (∏ j ∈ s.erase i, (sec j).2) • (sec i).1
  · simp only [LinearMap.map_smul_of_tower, Submonoid.coe_finset_prod]
    rw [← hsec, ← mul_smul, Submonoid.smul_def]
    congr
    simp only [Submonoid.coe_mul, Submonoid.coe_finset_prod, mul_comm]
    rw [← Finset.prod_insert (f := fun i ↦ ((sec i).snd).val) (s.not_mem_erase i),
      Finset.insert_erase hi]


/-- We can clear the denominators of a finite indexed family of fractions. -/
theorem exist_integer_multiples_of_finite {ι : Type*} [Finite ι] (g : ι → M') :
    ∃ b : S, ∀ i, IsInteger f ((b : R) • g i) := by
  /-
    R : Type u_1
    inst✝⁶ : CommSemiring R
    S : Submonoid R
    M : Type u_2
    inst✝⁵ : AddCommMonoid M
    inst✝⁴ : Module R M
    M' : Type u_3
    inst✝³ : AddCommMonoid M'
    inst✝² : Module R M'
    f : LinearMap (RingHom.id R) M M'
    inst✝¹ : IsLocalizedModule S f
    ι : Type u_4
    inst✝ : Finite ι
    g : ι → M'
    ⊢ Exists fun b => ∀ (i : ι), IsLocalizedModule.IsInteger f (HSMul.hSMul (↑b) ( …
  -/
  cases nonempty_fintype ι
  /-
    case intro
    R : Type u_1
    inst✝⁶ : CommSemiring R
    S : Submonoid R
    M : Type u_2
    inst✝⁵ : AddCommMonoid M
    inst✝⁴ : Module R M
    M' : Type u_3
    inst✝³ : AddCommMonoid M'
    inst✝² : Module R M'
    f : LinearMap (RingHom.id R) M M'
    inst✝¹ : IsLocalizedModule S f
    ι : Type u_4
    inst✝ : Finite ι
    g : ι → M'
    val✝ : Fintype ι
    ⊢ Exists fun b => ∀ (i : ι), IsLocalizedModule.IsInteger f (HSMul.hSMul (↑b) ( …
  -/
  obtain ⟨b, hb⟩ := exist_integer_multiples S f Finset.univ g
  /-
    case intro.intro
    R : Type u_1
    inst✝⁶ : CommSemiring R
    S : Submonoid R
    M : Type u_2
    inst✝⁵ : AddCommMonoid M
    inst✝⁴ : Module R M
    M' : Type u_3
    inst✝³ : AddCommMonoid M'
    inst✝² : Module R M'
    f : LinearMap (RingHom.id R) M M'
    inst✝¹ : IsLocalizedModule S f
    ι : Type u_4
    inst✝ : Finite ι
    g : ι → M'
    val✝ : Fintype ι
    b : Subtype fun x => Membership.mem S x
    hb : ∀ (i : ι), Membership.mem Finset.univ i → IsLocalizedModule.IsInteger f ( …
    ⊢ Exists fun b => ∀ (i : ι), IsLocalizedModule.IsInteger f (HSMul.hSMul (↑b) ( …
  -/
  exact ⟨b, fun i => hb i (Finset.mem_univ _)⟩
  /-
    🎉 no goals
  -/


/-- We can clear the denominators of a finite set of fractions. -/
theorem exist_integer_multiples_of_finset (s : Finset M') :
    ∃ b : S, ∀ a ∈ s, IsInteger f ((b : R) • a) :=
  exist_integer_multiples S f s id


/-- A choice of a common multiple of the denominators of a `Finset`-indexed family of fractions. -/
noncomputable def commonDenom {ι : Type*} (s : Finset ι) (g : ι → M') : S :=
  (exist_integer_multiples S f s g).choose


/-- The numerator of a fraction after clearing the denominators
of a `Finset`-indexed family of fractions. -/
noncomputable def integerMultiple {ι : Type*} (s : Finset ι) (g : ι → M') (i : s) : M :=
  ((exist_integer_multiples S f s g).choose_spec i i.prop).choose


@[simp]
theorem map_integerMultiple {ι : Type*} (s : Finset ι) (g : ι → M') (i : s) :
    f (integerMultiple S f s g i) = commonDenom S f s g • g i :=
  ((exist_integer_multiples S f s g).choose_spec _ i.prop).choose_spec


/-- A choice of a common multiple of the denominators of a finite set of fractions. -/
noncomputable def commonDenomOfFinset (s : Finset M') : S :=
  commonDenom S f s id


/-- The finset of numerators after clearing the denominators of a finite set of fractions. -/
noncomputable def finsetIntegerMultiple [DecidableEq M] (s : Finset M') : Finset M :=
  s.attach.image fun t => integerMultiple S f s id t


theorem finsetIntegerMultiple_image [DecidableEq M] (s : Finset M') :
    f '' finsetIntegerMultiple S f s = commonDenomOfFinset S f s • (s : Set M') := by
  /-
    R : Type u_1
    inst✝⁶ : CommSemiring R
    S : Submonoid R
    M : Type u_2
    inst✝⁵ : AddCommMonoid M
    inst✝⁴ : Module R M
    M' : Type u_3
    inst✝³ : AddCommMonoid M'
    inst✝² : Module R M'
    f : LinearMap (RingHom.id R) M M'
    inst✝¹ : IsLocalizedModule S f
    inst✝ : DecidableEq M
    s : Finset M'
    ⊢ Eq (Set.image ⇑f ↑(IsLocalizedModule.finsetIntegerMultiple S f s)) (HSMul.hS …
  -/
  delta finsetIntegerMultiple commonDenom
  /-
    R : Type u_1
    inst✝⁶ : CommSemiring R
    S : Submonoid R
    M : Type u_2
    inst✝⁵ : AddCommMonoid M
    inst✝⁴ : Module R M
    M' : Type u_3
    inst✝³ : AddCommMonoid M'
    inst✝² : Module R M'
    f : LinearMap (RingHom.id R) M M'
    inst✝¹ : IsLocalizedModule S f
    inst✝ : DecidableEq M
    s : Finset M'
    ⊢ Eq (Set.image ⇑f ↑(Finset.image (fun t => IsLocalizedModule.integerMultiple  …
  -/
  rw [Finset.coe_image]
  /-
    R : Type u_1
    inst✝⁶ : CommSemiring R
    S : Submonoid R
    M : Type u_2
    inst✝⁵ : AddCommMonoid M
    inst✝⁴ : Module R M
    M' : Type u_3
    inst✝³ : AddCommMonoid M'
    inst✝² : Module R M'
    f : LinearMap (RingHom.id R) M M'
    inst✝¹ : IsLocalizedModule S f
    inst✝ : DecidableEq M
    s : Finset M'
    ⊢ Eq (Set.image (⇑f) (Set.image (fun t => IsLocalizedModule.integerMultiple S  …
  -/
  ext
  /-
    case h
    R : Type u_1
    inst✝⁶ : CommSemiring R
    S : Submonoid R
    M : Type u_2
    inst✝⁵ : AddCommMonoid M
    inst✝⁴ : Module R M
    M' : Type u_3
    inst✝³ : AddCommMonoid M'
    inst✝² : Module R M'
    f : LinearMap (RingHom.id R) M M'
    inst✝¹ : IsLocalizedModule S f
    inst✝ : DecidableEq M
    s : Finset M'
    x✝ : M'
    ⊢ Iff (Membership.mem (Set.image (⇑f) (Set.image (fun t => IsLocalizedModule.i …
  -/
  constructor
    /-
      case h.mp
      R : Type u_1
      inst✝⁶ : CommSemiring R
      S : Submonoid R
      M : Type u_2
      inst✝⁵ : AddCommMonoid M
      inst✝⁴ : Module R M
      M' : Type u_3
      inst✝³ : AddCommMonoid M'
      inst✝² : Module R M'
      f : LinearMap (RingHom.id R) M M'
      inst✝¹ : IsLocalizedModule S f
      inst✝ : DecidableEq M
      s : Finset M'
      x✝ : M'
      ⊢ Membership.mem (Set.image (⇑f) (Set.image (fun t => IsLocalizedModule.intege …
    -/
  · rintro ⟨_, ⟨x, -, rfl⟩, rfl⟩
    /-
      case h.mp.intro.intro.intro.intro
      R : Type u_1
      inst✝⁶ : CommSemiring R
      S : Submonoid R
      M : Type u_2
      inst✝⁵ : AddCommMonoid M
      inst✝⁴ : Module R M
      M' : Type u_3
      inst✝³ : AddCommMonoid M'
      inst✝² : Module R M'
      f : LinearMap (RingHom.id R) M M'
      inst✝¹ : IsLocalizedModule S f
      inst✝ : DecidableEq M
      s : Finset M'
      x : Subtype fun x => Membership.mem s x
      ⊢ Membership.mem (HSMul.hSMul (IsLocalizedModule.commonDenomOfFinset S f s) ↑s …
    -/
    rw [map_integerMultiple]
    /-
      case h.mp.intro.intro.intro.intro
      R : Type u_1
      inst✝⁶ : CommSemiring R
      S : Submonoid R
      M : Type u_2
      inst✝⁵ : AddCommMonoid M
      inst✝⁴ : Module R M
      M' : Type u_3
      inst✝³ : AddCommMonoid M'
      inst✝² : Module R M'
      f : LinearMap (RingHom.id R) M M'
      inst✝¹ : IsLocalizedModule S f
      inst✝ : DecidableEq M
      s : Finset M'
      x : Subtype fun x => Membership.mem s x
      ⊢ Membership.mem (HSMul.hSMul (IsLocalizedModule.commonDenomOfFinset S f s) ↑s …
    -/
    exact Set.mem_image_of_mem _ x.prop
    /-
      🎉 no goals
    -/
    /-
      case h.mpr
      R : Type u_1
      inst✝⁶ : CommSemiring R
      S : Submonoid R
      M : Type u_2
      inst✝⁵ : AddCommMonoid M
      inst✝⁴ : Module R M
      M' : Type u_3
      inst✝³ : AddCommMonoid M'
      inst✝² : Module R M'
      f : LinearMap (RingHom.id R) M M'
      inst✝¹ : IsLocalizedModule S f
      inst✝ : DecidableEq M
      s : Finset M'
      x✝ : M'
      ⊢ Membership.mem (HSMul.hSMul (IsLocalizedModule.commonDenomOfFinset S f s) ↑s …
    -/
  · rintro ⟨x, hx, rfl⟩
    /-
      case h.mpr.intro.intro
      R : Type u_1
      inst✝⁶ : CommSemiring R
      S : Submonoid R
      M : Type u_2
      inst✝⁵ : AddCommMonoid M
      inst✝⁴ : Module R M
      M' : Type u_3
      inst✝³ : AddCommMonoid M'
      inst✝² : Module R M'
      f : LinearMap (RingHom.id R) M M'
      inst✝¹ : IsLocalizedModule S f
      inst✝ : DecidableEq M
      s : Finset M'
      x : M'
      hx : Membership.mem (↑s) x
      ⊢ Membership.mem (Set.image (⇑f) (Set.image (fun t => IsLocalizedModule.intege …
    -/
    exact ⟨_, ⟨⟨x, hx⟩, s.mem_attach _, rfl⟩, map_integerMultiple S f s id _⟩
    /-
      🎉 no goals
    -/


theorem smul_mem_finsetIntegerMultiple_span [DecidableEq M] (x : M) (s : Finset M')
    (hx : f x ∈ Submodule.span R s) :
    ∃ (m : S), m • x ∈ Submodule.span R (IsLocalizedModule.finsetIntegerMultiple S f s) := by
  /-
    R : Type u_1
    inst✝⁶ : CommSemiring R
    S : Submonoid R
    M : Type u_2
    inst✝⁵ : AddCommMonoid M
    inst✝⁴ : Module R M
    M' : Type u_3
    inst✝³ : AddCommMonoid M'
    inst✝² : Module R M'
    f : LinearMap (RingHom.id R) M M'
    inst✝¹ : IsLocalizedModule S f
    inst✝ : DecidableEq M
    x : M
    s : Finset M'
    hx : Membership.mem (Submodule.span R ↑s) (f x)
    ⊢ Exists fun m => Membership.mem (Submodule.span R ↑(IsLocalizedModule.finsetI …
  -/
  let y : S := IsLocalizedModule.commonDenomOfFinset S f s
  have hx₁ : (y : R) • (s : Set M') = f '' _ :=
    (IsLocalizedModule.finsetIntegerMultiple_image S f s).symm
  /-
    R : Type u_1
    inst✝⁶ : CommSemiring R
    S : Submonoid R
    M : Type u_2
    inst✝⁵ : AddCommMonoid M
    inst✝⁴ : Module R M
    M' : Type u_3
    inst✝³ : AddCommMonoid M'
    inst✝² : Module R M'
    f : LinearMap (RingHom.id R) M M'
    inst✝¹ : IsLocalizedModule S f
    inst✝ : DecidableEq M
    x : M
    s : Finset M'
    hx : Membership.mem (Submodule.span R ↑s) (f x)
    y : Subtype fun x => Membership.mem S x := IsLocalizedModule.commonDenomOfFins …
    hx₁ : Eq (HSMul.hSMul ↑y ↑s) (Set.image ⇑f ↑(IsLocalizedModule.finsetIntegerMu …
    ⊢ Exists fun m => Membership.mem (Submodule.span R ↑(IsLocalizedModule.finsetI …
  -/
  apply congrArg (Submodule.span R) at hx₁
  /-
    R : Type u_1
    inst✝⁶ : CommSemiring R
    S : Submonoid R
    M : Type u_2
    inst✝⁵ : AddCommMonoid M
    inst✝⁴ : Module R M
    M' : Type u_3
    inst✝³ : AddCommMonoid M'
    inst✝² : Module R M'
    f : LinearMap (RingHom.id R) M M'
    inst✝¹ : IsLocalizedModule S f
    inst✝ : DecidableEq M
    x : M
    s : Finset M'
    hx : Membership.mem (Submodule.span R ↑s) (f x)
    y : Subtype fun x => Membership.mem S x := IsLocalizedModule.commonDenomOfFins …
    hx₁ : Eq (Submodule.span R (HSMul.hSMul ↑y ↑s)) (Submodule.span R (Set.image ⇑ …
    ⊢ Exists fun m => Membership.mem (Submodule.span R ↑(IsLocalizedModule.finsetI …
  -/
  rw [Submodule.span_smul] at hx₁
  /-
    R : Type u_1
    inst✝⁶ : CommSemiring R
    S : Submonoid R
    M : Type u_2
    inst✝⁵ : AddCommMonoid M
    inst✝⁴ : Module R M
    M' : Type u_3
    inst✝³ : AddCommMonoid M'
    inst✝² : Module R M'
    f : LinearMap (RingHom.id R) M M'
    inst✝¹ : IsLocalizedModule S f
    inst✝ : DecidableEq M
    x : M
    s : Finset M'
    hx : Membership.mem (Submodule.span R ↑s) (f x)
    y : Subtype fun x => Membership.mem S x := IsLocalizedModule.commonDenomOfFins …
    hx₁ : Eq (HSMul.hSMul (↑y) (Submodule.span R ↑s)) (Submodule.span R (Set.image …
    ⊢ Exists fun m => Membership.mem (Submodule.span R ↑(IsLocalizedModule.finsetI …
  -/
  replace hx : _ ∈ y • Submodule.span R (s : Set M') := Set.smul_mem_smul_set hx
  /-
    R : Type u_1
    inst✝⁶ : CommSemiring R
    S : Submonoid R
    M : Type u_2
    inst✝⁵ : AddCommMonoid M
    inst✝⁴ : Module R M
    M' : Type u_3
    inst✝³ : AddCommMonoid M'
    inst✝² : Module R M'
    f : LinearMap (RingHom.id R) M M'
    inst✝¹ : IsLocalizedModule S f
    inst✝ : DecidableEq M
    x : M
    s : Finset M'
    y : Subtype fun x => Membership.mem S x := IsLocalizedModule.commonDenomOfFins …
    hx₁ : Eq (HSMul.hSMul (↑y) (Submodule.span R ↑s)) (Submodule.span R (Set.image …
    hx : Membership.mem (HSMul.hSMul y (Submodule.span R ↑s)) (HSMul.hSMul (S.subt …
    ⊢ Exists fun m => Membership.mem (Submodule.span R ↑(IsLocalizedModule.finsetI …
  -/
  erw [hx₁, ← f.map_smul, ← Submodule.map_span f] at hx
  /-
    R : Type u_1
    inst✝⁶ : CommSemiring R
    S : Submonoid R
    M : Type u_2
    inst✝⁵ : AddCommMonoid M
    inst✝⁴ : Module R M
    M' : Type u_3
    inst✝³ : AddCommMonoid M'
    inst✝² : Module R M'
    f : LinearMap (RingHom.id R) M M'
    inst✝¹ : IsLocalizedModule S f
    inst✝ : DecidableEq M
    x : M
    s : Finset M'
    y : Subtype fun x => Membership.mem S x := IsLocalizedModule.commonDenomOfFins …
    hx₁ : Eq (HSMul.hSMul (↑y) (Submodule.span R ↑s)) (Submodule.span R (Set.image …
    hx : Membership.mem (Submodule.map f (Submodule.span R ↑(IsLocalizedModule.fin …
    ⊢ Exists fun m => Membership.mem (Submodule.span R ↑(IsLocalizedModule.finsetI …
  -/
  obtain ⟨x', hx', hx''⟩ := hx
  /-
    case intro.intro
    R : Type u_1
    inst✝⁶ : CommSemiring R
    S : Submonoid R
    M : Type u_2
    inst✝⁵ : AddCommMonoid M
    inst✝⁴ : Module R M
    M' : Type u_3
    inst✝³ : AddCommMonoid M'
    inst✝² : Module R M'
    f : LinearMap (RingHom.id R) M M'
    inst✝¹ : IsLocalizedModule S f
    inst✝ : DecidableEq M
    x : M
    s : Finset M'
    y : Subtype fun x => Membership.mem S x := IsLocalizedModule.commonDenomOfFins …
    hx₁ : Eq (HSMul.hSMul (↑y) (Submodule.span R ↑s)) (Submodule.span R (Set.image …
    x' : M
    hx' : Membership.mem (↑(Submodule.span R ↑(IsLocalizedModule.finsetIntegerMult …
    hx'' : Eq (f x') (f (HSMul.hSMul (S.subtype y) x))
    ⊢ Exists fun m => Membership.mem (Submodule.span R ↑(IsLocalizedModule.finsetI …
  -/
  obtain ⟨a, ha⟩ := (IsLocalizedModule.eq_iff_exists S f).mp hx''
  /-
    case intro.intro.intro
    R : Type u_1
    inst✝⁶ : CommSemiring R
    S : Submonoid R
    M : Type u_2
    inst✝⁵ : AddCommMonoid M
    inst✝⁴ : Module R M
    M' : Type u_3
    inst✝³ : AddCommMonoid M'
    inst✝² : Module R M'
    f : LinearMap (RingHom.id R) M M'
    inst✝¹ : IsLocalizedModule S f
    inst✝ : DecidableEq M
    x : M
    s : Finset M'
    y : Subtype fun x => Membership.mem S x := IsLocalizedModule.commonDenomOfFins …
    hx₁ : Eq (HSMul.hSMul (↑y) (Submodule.span R ↑s)) (Submodule.span R (Set.image …
    x' : M
    hx' : Membership.mem (↑(Submodule.span R ↑(IsLocalizedModule.finsetIntegerMult …
    hx'' : Eq (f x') (f (HSMul.hSMul (S.subtype y) x))
    a : Subtype fun x => Membership.mem S x
    ha : Eq (HSMul.hSMul a x') (HSMul.hSMul a (HSMul.hSMul (S.subtype y) x))
    ⊢ Exists fun m => Membership.mem (Submodule.span R ↑(IsLocalizedModule.finsetI …
  -/
  use a * y
  convert (Submodule.span R
    (IsLocalizedModule.finsetIntegerMultiple S f s : Set M)).smul_mem
      a hx' using 1
  /-
    case h.e'_5
    R : Type u_1
    inst✝⁶ : CommSemiring R
    S : Submonoid R
    M : Type u_2
    inst✝⁵ : AddCommMonoid M
    inst✝⁴ : Module R M
    M' : Type u_3
    inst✝³ : AddCommMonoid M'
    inst✝² : Module R M'
    f : LinearMap (RingHom.id R) M M'
    inst✝¹ : IsLocalizedModule S f
    inst✝ : DecidableEq M
    x : M
    s : Finset M'
    y : Subtype fun x => Membership.mem S x := IsLocalizedModule.commonDenomOfFins …
    hx₁ : Eq (HSMul.hSMul (↑y) (Submodule.span R ↑s)) (Submodule.span R (Set.image …
    x' : M
    hx' : Membership.mem (↑(Submodule.span R ↑(IsLocalizedModule.finsetIntegerMult …
    hx'' : Eq (f x') (f (HSMul.hSMul (S.subtype y) x))
    a : Subtype fun x => Membership.mem S x
    ha : Eq (HSMul.hSMul a x') (HSMul.hSMul a (HSMul.hSMul (S.subtype y) x))
    ⊢ Eq (HSMul.hSMul (HMul.hMul a y) x) (HSMul.hSMul (↑a) x')
  -/
  convert ha.symm using 1
  /-
    case h.e'_2
    R : Type u_1
    inst✝⁶ : CommSemiring R
    S : Submonoid R
    M : Type u_2
    inst✝⁵ : AddCommMonoid M
    inst✝⁴ : Module R M
    M' : Type u_3
    inst✝³ : AddCommMonoid M'
    inst✝² : Module R M'
    f : LinearMap (RingHom.id R) M M'
    inst✝¹ : IsLocalizedModule S f
    inst✝ : DecidableEq M
    x : M
    s : Finset M'
    y : Subtype fun x => Membership.mem S x := IsLocalizedModule.commonDenomOfFins …
    hx₁ : Eq (HSMul.hSMul (↑y) (Submodule.span R ↑s)) (Submodule.span R (Set.image …
    x' : M
    hx' : Membership.mem (↑(Submodule.span R ↑(IsLocalizedModule.finsetIntegerMult …
    hx'' : Eq (f x') (f (HSMul.hSMul (S.subtype y) x))
    a : Subtype fun x => Membership.mem S x
    ha : Eq (HSMul.hSMul a x') (HSMul.hSMul a (HSMul.hSMul (S.subtype y) x))
    ⊢ Eq (HSMul.hSMul (HMul.hMul a y) x) (HSMul.hSMul a (HSMul.hSMul (S.subtype y) …
  -/
  simp only [Submonoid.coe_subtype, Submonoid.smul_def]
  /-
    case h.e'_2
    R : Type u_1
    inst✝⁶ : CommSemiring R
    S : Submonoid R
    M : Type u_2
    inst✝⁵ : AddCommMonoid M
    inst✝⁴ : Module R M
    M' : Type u_3
    inst✝³ : AddCommMonoid M'
    inst✝² : Module R M'
    f : LinearMap (RingHom.id R) M M'
    inst✝¹ : IsLocalizedModule S f
    inst✝ : DecidableEq M
    x : M
    s : Finset M'
    y : Subtype fun x => Membership.mem S x := IsLocalizedModule.commonDenomOfFins …
    hx₁ : Eq (HSMul.hSMul (↑y) (Submodule.span R ↑s)) (Submodule.span R (Set.image …
    x' : M
    hx' : Membership.mem (↑(Submodule.span R ↑(IsLocalizedModule.finsetIntegerMult …
    hx'' : Eq (f x') (f (HSMul.hSMul (S.subtype y) x))
    a : Subtype fun x => Membership.mem S x
    ha : Eq (HSMul.hSMul a x') (HSMul.hSMul a (HSMul.hSMul (S.subtype y) x))
    ⊢ Eq (HSMul.hSMul (↑(HMul.hMul a y)) x) (HSMul.hSMul (↑a) (HSMul.hSMul (↑y) x))
  -/
  erw [← smul_smul]
  /-
    🎉 no goals
  -/


