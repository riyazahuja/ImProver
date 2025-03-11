/-- Given `a : S`, `S` a localization of `R`, `IsInteger R a` iff `a` is in the image of
the localization map from `R` to `S`. -/
def IsInteger (a : S) : Prop :=
  a ∈ (algebraMap R S).rangeS


theorem isInteger_zero : IsInteger R (0 : S) :=
  Subsemiring.zero_mem _


theorem isInteger_one : IsInteger R (1 : S) :=
  Subsemiring.one_mem _


theorem isInteger_add {a b : S} (ha : IsInteger R a) (hb : IsInteger R b) : IsInteger R (a + b) :=
  Subsemiring.add_mem _ ha hb


theorem isInteger_mul {a b : S} (ha : IsInteger R a) (hb : IsInteger R b) : IsInteger R (a * b) :=
  Subsemiring.mul_mem _ ha hb


theorem isInteger_smul {a : R} {b : S} (hb : IsInteger R b) : IsInteger R (a • b) := by
  /-
    R : Type u_1
    inst✝² : CommSemiring R
    S : Type u_2
    inst✝¹ : CommSemiring S
    inst✝ : Algebra R S
    a : R
    b : S
    hb : IsLocalization.IsInteger R b
    ⊢ IsLocalization.IsInteger R (HSMul.hSMul a b)
  -/
  rcases hb with ⟨b', hb⟩
  /-
    case intro
    R : Type u_1
    inst✝² : CommSemiring R
    S : Type u_2
    inst✝¹ : CommSemiring S
    inst✝ : Algebra R S
    a : R
    b : S
    b' : R
    hb : Eq ((algebraMap R S) b') b
    ⊢ IsLocalization.IsInteger R (HSMul.hSMul a b)
  -/
  use a * b'
  /-
    case h
    R : Type u_1
    inst✝² : CommSemiring R
    S : Type u_2
    inst✝¹ : CommSemiring S
    inst✝ : Algebra R S
    a : R
    b : S
    b' : R
    hb : Eq ((algebraMap R S) b') b
    ⊢ Eq ((algebraMap R S) (HMul.hMul a b')) (HSMul.hSMul a b)
  -/
  rw [← hb, (algebraMap R S).map_mul, Algebra.smul_def]
  /-
    🎉 no goals
  -/


/-- Each element `a : S` has an `M`-multiple which is an integer.

This version multiplies `a` on the right, matching the argument order in `LocalizationMap.surj`.
-/
theorem exists_integer_multiple' (a : S) : ∃ b : M, IsInteger R (a * algebraMap R S b) :=
  let ⟨⟨Num, denom⟩, h⟩ := IsLocalization.surj _ a
  ⟨denom, Set.mem_range.mpr ⟨Num, h.symm⟩⟩


/-- Each element `a : S` has an `M`-multiple which is an integer.

This version multiplies `a` on the left, matching the argument order in the `SMul` instance.
-/
theorem exists_integer_multiple (a : S) : ∃ b : M, IsInteger R ((b : R) • a) := by
  /-
    R : Type u_1
    inst✝³ : CommSemiring R
    M : Submonoid R
    S : Type u_2
    inst✝² : CommSemiring S
    inst✝¹ : Algebra R S
    inst✝ : IsLocalization M S
    a : S
    ⊢ Exists fun b => IsLocalization.IsInteger R (HSMul.hSMul (↑b) a)
  -/
  simp_rw [Algebra.smul_def, mul_comm _ a]
  /-
    R : Type u_1
    inst✝³ : CommSemiring R
    M : Submonoid R
    S : Type u_2
    inst✝² : CommSemiring S
    inst✝¹ : Algebra R S
    inst✝ : IsLocalization M S
    a : S
    ⊢ Exists fun b => IsLocalization.IsInteger R (HMul.hMul a ((algebraMap R S) ↑b))
  -/
  apply exists_integer_multiple'
  /-
    🎉 no goals
  -/


/-- We can clear the denominators of a `Finset`-indexed family of fractions. -/
theorem exist_integer_multiples {ι : Type*} (s : Finset ι) (f : ι → S) :
    ∃ b : M, ∀ i ∈ s, IsLocalization.IsInteger R ((b : R) • f i) := by
  /-
    R : Type u_1
    inst✝³ : CommSemiring R
    M : Submonoid R
    S : Type u_2
    inst✝² : CommSemiring S
    inst✝¹ : Algebra R S
    inst✝ : IsLocalization M S
    ι : Type u_4
    s : Finset ι
    f : ι → S
    ⊢ Exists fun b => ∀ (i : ι), Membership.mem s i → IsLocalization.IsInteger R ( …
  -/
  haveI := Classical.propDecidable
  /-
    R : Type u_1
    inst✝³ : CommSemiring R
    M : Submonoid R
    S : Type u_2
    inst✝² : CommSemiring S
    inst✝¹ : Algebra R S
    inst✝ : IsLocalization M S
    ι : Type u_4
    s : Finset ι
    f : ι → S
    this : (a : Prop) → Decidable a
    ⊢ Exists fun b => ∀ (i : ι), Membership.mem s i → IsLocalization.IsInteger R ( …
  -/
  refine ⟨∏ i ∈ s, (sec M (f i)).2, fun i hi => ⟨?_, ?_⟩⟩
    /-
      case refine_1
      R : Type u_1
      inst✝³ : CommSemiring R
      M : Submonoid R
      S : Type u_2
      inst✝² : CommSemiring S
      inst✝¹ : Algebra R S
      inst✝ : IsLocalization M S
      ι : Type u_4
      s : Finset ι
      f : ι → S
      this : (a : Prop) → Decidable a
      i : ι
      hi : Membership.mem s i
      ⊢ R
    -/
  · exact (∏ j ∈ s.erase i, (sec M (f j)).2) * (sec M (f i)).1
    /-
      🎉 no goals
    -/
  /-
    case refine_2
    R : Type u_1
    inst✝³ : CommSemiring R
    M : Submonoid R
    S : Type u_2
    inst✝² : CommSemiring S
    inst✝¹ : Algebra R S
    inst✝ : IsLocalization M S
    ι : Type u_4
    s : Finset ι
    f : ι → S
    this : (a : Prop) → Decidable a
    i : ι
    hi : Membership.mem s i
    ⊢ Eq ((algebraMap R S) (HMul.hMul (↑((s.erase i).prod fun j => (IsLocalization …
  -/
  rw [RingHom.map_mul, sec_spec', ← mul_assoc, ← (algebraMap R S).map_mul, ← Algebra.smul_def]
  /-
    case refine_2
    R : Type u_1
    inst✝³ : CommSemiring R
    M : Submonoid R
    S : Type u_2
    inst✝² : CommSemiring S
    inst✝¹ : Algebra R S
    inst✝ : IsLocalization M S
    ι : Type u_4
    s : Finset ι
    f : ι → S
    this : (a : Prop) → Decidable a
    i : ι
    hi : Membership.mem s i
    ⊢ Eq (HSMul.hSMul (HMul.hMul ↑((s.erase i).prod fun j => (IsLocalization.sec M …
  -/
  congr 2
  /-
    case refine_2.e_a
    R : Type u_1
    inst✝³ : CommSemiring R
    M : Submonoid R
    S : Type u_2
    inst✝² : CommSemiring S
    inst✝¹ : Algebra R S
    inst✝ : IsLocalization M S
    ι : Type u_4
    s : Finset ι
    f : ι → S
    this : (a : Prop) → Decidable a
    i : ι
    hi : Membership.mem s i
    ⊢ Eq (HMul.hMul ↑((s.erase i).prod fun j => (IsLocalization.sec M (f j)).2) ↑( …
  -/
  refine _root_.trans ?_ (map_prod (Submonoid.subtype M) _ _).symm
  rw [mul_comm,Submonoid.coe_finset_prod,
    -- Porting note: explicitly supplied `f`
    ← Finset.prod_insert (f := fun i => ((sec M (f i)).snd : R)) (s.not_mem_erase i),
    Finset.insert_erase hi]
  /-
    case refine_2.e_a
    R : Type u_1
    inst✝³ : CommSemiring R
    M : Submonoid R
    S : Type u_2
    inst✝² : CommSemiring S
    inst✝¹ : Algebra R S
    inst✝ : IsLocalization M S
    ι : Type u_4
    s : Finset ι
    f : ι → S
    this : (a : Prop) → Decidable a
    i : ι
    hi : Membership.mem s i
    ⊢ Eq (s.prod fun x => ↑(IsLocalization.sec M (f x)).2) (s.prod fun x => M.subt …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- We can clear the denominators of a finite indexed family of fractions. -/
theorem exist_integer_multiples_of_finite {ι : Type*} [Finite ι] (f : ι → S) :
    ∃ b : M, ∀ i, IsLocalization.IsInteger R ((b : R) • f i) := by
  /-
    R : Type u_1
    inst✝⁴ : CommSemiring R
    M : Submonoid R
    S : Type u_2
    inst✝³ : CommSemiring S
    inst✝² : Algebra R S
    inst✝¹ : IsLocalization M S
    ι : Type u_4
    inst✝ : Finite ι
    f : ι → S
    ⊢ Exists fun b => ∀ (i : ι), IsLocalization.IsInteger R (HSMul.hSMul (↑b) (f i))
  -/
  cases nonempty_fintype ι
  /-
    case intro
    R : Type u_1
    inst✝⁴ : CommSemiring R
    M : Submonoid R
    S : Type u_2
    inst✝³ : CommSemiring S
    inst✝² : Algebra R S
    inst✝¹ : IsLocalization M S
    ι : Type u_4
    inst✝ : Finite ι
    f : ι → S
    val✝ : Fintype ι
    ⊢ Exists fun b => ∀ (i : ι), IsLocalization.IsInteger R (HSMul.hSMul (↑b) (f i))
  -/
  obtain ⟨b, hb⟩ := exist_integer_multiples M Finset.univ f
  /-
    case intro.intro
    R : Type u_1
    inst✝⁴ : CommSemiring R
    M : Submonoid R
    S : Type u_2
    inst✝³ : CommSemiring S
    inst✝² : Algebra R S
    inst✝¹ : IsLocalization M S
    ι : Type u_4
    inst✝ : Finite ι
    f : ι → S
    val✝ : Fintype ι
    b : Subtype fun x => Membership.mem M x
    hb : ∀ (i : ι), Membership.mem Finset.univ i → IsLocalization.IsInteger R (HSM …
    ⊢ Exists fun b => ∀ (i : ι), IsLocalization.IsInteger R (HSMul.hSMul (↑b) (f i))
  -/
  exact ⟨b, fun i => hb i (Finset.mem_univ _)⟩
  /-
    🎉 no goals
  -/


/-- We can clear the denominators of a finite set of fractions. -/
theorem exist_integer_multiples_of_finset (s : Finset S) :
    ∃ b : M, ∀ a ∈ s, IsInteger R ((b : R) • a) :=
  exist_integer_multiples M s id


/-- A choice of a common multiple of the denominators of a `Finset`-indexed family of fractions. -/
noncomputable def commonDenom {ι : Type*} (s : Finset ι) (f : ι → S) : M :=
  (exist_integer_multiples M s f).choose


/-- The numerator of a fraction after clearing the denominators
of a `Finset`-indexed family of fractions. -/
noncomputable def integerMultiple {ι : Type*} (s : Finset ι) (f : ι → S) (i : s) : R :=
  ((exist_integer_multiples M s f).choose_spec i i.prop).choose


@[simp]
theorem map_integerMultiple {ι : Type*} (s : Finset ι) (f : ι → S) (i : s) :
    algebraMap R S (integerMultiple M s f i) = commonDenom M s f • f i :=
  ((exist_integer_multiples M s f).choose_spec _ i.prop).choose_spec


/-- A choice of a common multiple of the denominators of a finite set of fractions. -/
noncomputable def commonDenomOfFinset (s : Finset S) : M :=
  commonDenom M s id


/-- The finset of numerators after clearing the denominators of a finite set of fractions. -/
noncomputable def finsetIntegerMultiple [DecidableEq R] (s : Finset S) : Finset R :=
  s.attach.image fun t => integerMultiple M s id t


theorem finsetIntegerMultiple_image [DecidableEq R] (s : Finset S) :
    algebraMap R S '' finsetIntegerMultiple M s = commonDenomOfFinset M s • (s : Set S) := by
  /-
    R : Type u_1
    inst✝⁴ : CommSemiring R
    M : Submonoid R
    S : Type u_2
    inst✝³ : CommSemiring S
    inst✝² : Algebra R S
    inst✝¹ : IsLocalization M S
    inst✝ : DecidableEq R
    s : Finset S
    ⊢ Eq (Set.image ⇑(algebraMap R S) ↑(IsLocalization.finsetIntegerMultiple M s)) …
  -/
  delta finsetIntegerMultiple commonDenom
  /-
    R : Type u_1
    inst✝⁴ : CommSemiring R
    M : Submonoid R
    S : Type u_2
    inst✝³ : CommSemiring S
    inst✝² : Algebra R S
    inst✝¹ : IsLocalization M S
    inst✝ : DecidableEq R
    s : Finset S
    ⊢ Eq (Set.image ⇑(algebraMap R S) ↑(Finset.image (fun t => IsLocalization.inte …
  -/
  rw [Finset.coe_image]
  /-
    R : Type u_1
    inst✝⁴ : CommSemiring R
    M : Submonoid R
    S : Type u_2
    inst✝³ : CommSemiring S
    inst✝² : Algebra R S
    inst✝¹ : IsLocalization M S
    inst✝ : DecidableEq R
    s : Finset S
    ⊢ Eq (Set.image (⇑(algebraMap R S)) (Set.image (fun t => IsLocalization.intege …
  -/
  ext
  /-
    case h
    R : Type u_1
    inst✝⁴ : CommSemiring R
    M : Submonoid R
    S : Type u_2
    inst✝³ : CommSemiring S
    inst✝² : Algebra R S
    inst✝¹ : IsLocalization M S
    inst✝ : DecidableEq R
    s : Finset S
    x✝ : S
    ⊢ Iff (Membership.mem (Set.image (⇑(algebraMap R S)) (Set.image (fun t => IsLo …
  -/
  constructor
    /-
      case h.mp
      R : Type u_1
      inst✝⁴ : CommSemiring R
      M : Submonoid R
      S : Type u_2
      inst✝³ : CommSemiring S
      inst✝² : Algebra R S
      inst✝¹ : IsLocalization M S
      inst✝ : DecidableEq R
      s : Finset S
      x✝ : S
      ⊢ Membership.mem (Set.image (⇑(algebraMap R S)) (Set.image (fun t => IsLocaliz …
    -/
  · rintro ⟨_, ⟨x, -, rfl⟩, rfl⟩
    /-
      case h.mp.intro.intro.intro.intro
      R : Type u_1
      inst✝⁴ : CommSemiring R
      M : Submonoid R
      S : Type u_2
      inst✝³ : CommSemiring S
      inst✝² : Algebra R S
      inst✝¹ : IsLocalization M S
      inst✝ : DecidableEq R
      s : Finset S
      x : Subtype fun x => Membership.mem s x
      ⊢ Membership.mem (HSMul.hSMul (IsLocalization.commonDenomOfFinset M s) ↑s) ((a …
    -/
    rw [map_integerMultiple]
    /-
      case h.mp.intro.intro.intro.intro
      R : Type u_1
      inst✝⁴ : CommSemiring R
      M : Submonoid R
      S : Type u_2
      inst✝³ : CommSemiring S
      inst✝² : Algebra R S
      inst✝¹ : IsLocalization M S
      inst✝ : DecidableEq R
      s : Finset S
      x : Subtype fun x => Membership.mem s x
      ⊢ Membership.mem (HSMul.hSMul (IsLocalization.commonDenomOfFinset M s) ↑s) (HS …
    -/
    exact Set.mem_image_of_mem _ x.prop
    /-
      🎉 no goals
    -/
    /-
      case h.mpr
      R : Type u_1
      inst✝⁴ : CommSemiring R
      M : Submonoid R
      S : Type u_2
      inst✝³ : CommSemiring S
      inst✝² : Algebra R S
      inst✝¹ : IsLocalization M S
      inst✝ : DecidableEq R
      s : Finset S
      x✝ : S
      ⊢ Membership.mem (HSMul.hSMul (IsLocalization.commonDenomOfFinset M s) ↑s) x✝  …
    -/
  · rintro ⟨x, hx, rfl⟩
    /-
      case h.mpr.intro.intro
      R : Type u_1
      inst✝⁴ : CommSemiring R
      M : Submonoid R
      S : Type u_2
      inst✝³ : CommSemiring S
      inst✝² : Algebra R S
      inst✝¹ : IsLocalization M S
      inst✝ : DecidableEq R
      s : Finset S
      x : S
      hx : Membership.mem (↑s) x
      ⊢ Membership.mem (Set.image (⇑(algebraMap R S)) (Set.image (fun t => IsLocaliz …
    -/
    exact ⟨_, ⟨⟨x, hx⟩, s.mem_attach _, rfl⟩, map_integerMultiple M s id _⟩
    /-
      🎉 no goals
    -/


