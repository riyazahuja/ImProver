/--
A ring of characteristic zero is of "mixed characteristic `(0, p)`" if there exists an ideal
such that the quotient `R ⧸ I` has characteristic `p`.

**Remark:** For `p = 0`, `MixedChar R 0` is a meaningless definition (i.e. satisfied by any ring)
as `R ⧸ ⊥ ≅ R` has by definition always characteristic zero.
One could require `(I ≠ ⊥)` in the definition, but then `MixedChar R 0` would mean something
like `ℤ`-algebra of extension degree `≥ 1` and would be completely independent from
whether something is a `ℚ`-algebra or not (e.g. `ℚ[X]` would satisfy it but `ℚ` wouldn't).
-/
class MixedCharZero (p : ℕ) : Prop where
  [toCharZero : CharZero R]
  charP_quotient : ∃ I : Ideal R, I ≠ ⊤ ∧ CharP (R ⧸ I) p


/--
Reduction to `p` prime: When proving any statement `P` about mixed characteristic rings we
can always assume that `p` is prime.
-/
theorem reduce_to_p_prime {P : Prop} :
    (∀ p > 0, MixedCharZero R p → P) ↔ ∀ p : ℕ, p.Prime → MixedCharZero R p → P := by
  /-
    R : Type u_1
    inst✝ : CommRing R
    P : Prop
    ⊢ Iff (∀ (p : Nat), GT.gt p 0 → MixedCharZero R p → P) (∀ (p : Nat), Nat.Prime …
  -/
  constructor
    /-
      case mp
      R : Type u_1
      inst✝ : CommRing R
      P : Prop
      ⊢ (∀ (p : Nat), GT.gt p 0 → MixedCharZero R p → P) → ∀ (p : Nat), Nat.Prime p  …
    -/
  · intro h q q_prime q_mixedChar
    /-
      case mp
      R : Type u_1
      inst✝ : CommRing R
      P : Prop
      h : ∀ (p : Nat), GT.gt p 0 → MixedCharZero R p → P
      q : Nat
      q_prime : Nat.Prime q
      q_mixedChar : MixedCharZero R q
      ⊢ P
    -/
    exact h q (Nat.Prime.pos q_prime) q_mixedChar
    /-
      🎉 no goals
    -/
    /-
      case mpr
      R : Type u_1
      inst✝ : CommRing R
      P : Prop
      ⊢ (∀ (p : Nat), Nat.Prime p → MixedCharZero R p → P) → ∀ (p : Nat), GT.gt p 0  …
    -/
  · intro h q q_pos q_mixedChar
    /-
      case mpr
      R : Type u_1
      inst✝ : CommRing R
      P : Prop
      h : ∀ (p : Nat), Nat.Prime p → MixedCharZero R p → P
      q : Nat
      q_pos : GT.gt q 0
      q_mixedChar : MixedCharZero R q
      ⊢ P
    -/
    rcases q_mixedChar.charP_quotient with ⟨I, hI_ne_top, _⟩
    -- Krull's Thm: There exists a prime ideal `P` such that `I ≤ P`
    /-
      case mpr.intro.intro
      R : Type u_1
      inst✝ : CommRing R
      P : Prop
      h : ∀ (p : Nat), Nat.Prime p → MixedCharZero R p → P
      q : Nat
      q_pos : GT.gt q 0
      q_mixedChar : MixedCharZero R q
      I : Ideal R
      hI_ne_top : Ne I Top.top
      right✝ : CharP (HasQuotient.Quotient R I) q
      ⊢ P
    -/
    rcases Ideal.exists_le_maximal I hI_ne_top with ⟨M, hM_max, h_IM⟩
    /-
      case mpr.intro.intro.intro.intro
      R : Type u_1
      inst✝ : CommRing R
      P : Prop
      h : ∀ (p : Nat), Nat.Prime p → MixedCharZero R p → P
      q : Nat
      q_pos : GT.gt q 0
      q_mixedChar : MixedCharZero R q
      I : Ideal R
      hI_ne_top : Ne I Top.top
      right✝ : CharP (HasQuotient.Quotient R I) q
      M : Ideal R
      hM_max : M.IsMaximal
      h_IM : LE.le I M
      ⊢ P
    -/
    let r := ringChar (R ⧸ M)
    have r_pos : r ≠ 0 := by
      have q_zero :=
        congr_arg (Ideal.Quotient.factor I M h_IM) (CharP.cast_eq_zero (R ⧸ I) q)
      simp only [map_natCast, map_zero] at q_zero
      apply ne_zero_of_dvd_ne_zero (ne_of_gt q_pos)
      exact (CharP.cast_eq_zero_iff (R ⧸ M) r q).mp q_zero
    have r_prime : Nat.Prime r :=
      or_iff_not_imp_right.1 (CharP.char_is_prime_or_zero (R ⧸ M) r) r_pos
    /-
      case mpr.intro.intro.intro.intro
      R : Type u_1
      inst✝ : CommRing R
      P : Prop
      h : ∀ (p : Nat), Nat.Prime p → MixedCharZero R p → P
      q : Nat
      q_pos : GT.gt q 0
      q_mixedChar : MixedCharZero R q
      I : Ideal R
      hI_ne_top : Ne I Top.top
      right✝ : CharP (HasQuotient.Quotient R I) q
      M : Ideal R
      hM_max : M.IsMaximal
      h_IM : LE.le I M
      r : Nat := ringChar (HasQuotient.Quotient R M)
      r_pos : Ne r 0
      r_prime : Nat.Prime r
      ⊢ P
    -/
    apply h r r_prime
    /-
      case mpr.intro.intro.intro.intro
      R : Type u_1
      inst✝ : CommRing R
      P : Prop
      h : ∀ (p : Nat), Nat.Prime p → MixedCharZero R p → P
      q : Nat
      q_pos : GT.gt q 0
      q_mixedChar : MixedCharZero R q
      I : Ideal R
      hI_ne_top : Ne I Top.top
      right✝ : CharP (HasQuotient.Quotient R I) q
      M : Ideal R
      hM_max : M.IsMaximal
      h_IM : LE.le I M
      r : Nat := ringChar (HasQuotient.Quotient R M)
      r_pos : Ne r 0
      r_prime : Nat.Prime r
      ⊢ MixedCharZero R r
    -/
    have : CharZero R := q_mixedChar.toCharZero
    /-
      case mpr.intro.intro.intro.intro
      R : Type u_1
      inst✝ : CommRing R
      P : Prop
      h : ∀ (p : Nat), Nat.Prime p → MixedCharZero R p → P
      q : Nat
      q_pos : GT.gt q 0
      q_mixedChar : MixedCharZero R q
      I : Ideal R
      hI_ne_top : Ne I Top.top
      right✝ : CharP (HasQuotient.Quotient R I) q
      M : Ideal R
      hM_max : M.IsMaximal
      h_IM : LE.le I M
      r : Nat := ringChar (HasQuotient.Quotient R M)
      r_pos : Ne r 0
      r_prime : Nat.Prime r
      this : CharZero R
      ⊢ MixedCharZero R r
    -/
    exact ⟨⟨M, hM_max.ne_top, ringChar.of_eq rfl⟩⟩
    /-
      🎉 no goals
    -/


/--
Reduction to `I` prime ideal: When proving statements about mixed characteristic rings,
after we reduced to `p` prime, we can assume that the ideal `I` in the definition is maximal.
-/
theorem reduce_to_maximal_ideal {p : ℕ} (hp : Nat.Prime p) :
    (∃ I : Ideal R, I ≠ ⊤ ∧ CharP (R ⧸ I) p) ↔ ∃ I : Ideal R, I.IsMaximal ∧ CharP (R ⧸ I) p := by
  /-
    R : Type u_1
    inst✝ : CommRing R
    p : Nat
    hp : Nat.Prime p
    ⊢ Iff (Exists fun I => And (Ne I Top.top) (CharP (HasQuotient.Quotient R I) p) …
  -/
  constructor
    /-
      case mp
      R : Type u_1
      inst✝ : CommRing R
      p : Nat
      hp : Nat.Prime p
      ⊢ (Exists fun I => And (Ne I Top.top) (CharP (HasQuotient.Quotient R I) p)) →  …
    -/
  · intro g
    /-
      case mp
      R : Type u_1
      inst✝ : CommRing R
      p : Nat
      hp : Nat.Prime p
      g : Exists fun I => And (Ne I Top.top) (CharP (HasQuotient.Quotient R I) p)
      ⊢ Exists fun I => And I.IsMaximal (CharP (HasQuotient.Quotient R I) p)
    -/
    rcases g with ⟨I, ⟨hI_not_top, _⟩⟩
    -- Krull's Thm: There exists a prime ideal `M` such that `I ≤ M`.
    /-
      case mp.intro.intro
      R : Type u_1
      inst✝ : CommRing R
      p : Nat
      hp : Nat.Prime p
      I : Ideal R
      hI_not_top : Ne I Top.top
      right✝ : CharP (HasQuotient.Quotient R I) p
      ⊢ Exists fun I => And I.IsMaximal (CharP (HasQuotient.Quotient R I) p)
    -/
    rcases Ideal.exists_le_maximal I hI_not_top with ⟨M, ⟨hM_max, hM_ge⟩⟩
    /-
      case mp.intro.intro.intro.intro
      R : Type u_1
      inst✝ : CommRing R
      p : Nat
      hp : Nat.Prime p
      I : Ideal R
      hI_not_top : Ne I Top.top
      right✝ : CharP (HasQuotient.Quotient R I) p
      M : Ideal R
      hM_max : M.IsMaximal
      hM_ge : LE.le I M
      ⊢ Exists fun I => And I.IsMaximal (CharP (HasQuotient.Quotient R I) p)
    -/
    use M
    /-
      case h
      R : Type u_1
      inst✝ : CommRing R
      p : Nat
      hp : Nat.Prime p
      I : Ideal R
      hI_not_top : Ne I Top.top
      right✝ : CharP (HasQuotient.Quotient R I) p
      M : Ideal R
      hM_max : M.IsMaximal
      hM_ge : LE.le I M
      ⊢ And M.IsMaximal (CharP (HasQuotient.Quotient R M) p)
    -/
    constructor
      /-
        case h.left
        R : Type u_1
        inst✝ : CommRing R
        p : Nat
        hp : Nat.Prime p
        I : Ideal R
        hI_not_top : Ne I Top.top
        right✝ : CharP (HasQuotient.Quotient R I) p
        M : Ideal R
        hM_max : M.IsMaximal
        hM_ge : LE.le I M
        ⊢ M.IsMaximal
      -/
    · exact hM_max
      /-
        🎉 no goals
      -/
    · cases CharP.exists (R ⧸ M) with
      | intro r hr =>
        convert hr
        have r_dvd_p : r ∣ p := by
          rw [← CharP.cast_eq_zero_iff (R ⧸ M) r p]
          convert congr_arg (Ideal.Quotient.factor I M hM_ge) (CharP.cast_eq_zero (R ⧸ I) p)
        symm
        apply (Nat.Prime.eq_one_or_self_of_dvd hp r r_dvd_p).resolve_left
        exact CharP.char_ne_one (R ⧸ M) r
    /-
      case mpr
      R : Type u_1
      inst✝ : CommRing R
      p : Nat
      hp : Nat.Prime p
      ⊢ (Exists fun I => And I.IsMaximal (CharP (HasQuotient.Quotient R I) p)) → Exi …
    -/
  · intro ⟨I, hI_max, h_charP⟩
    /-
      case mpr
      R : Type u_1
      inst✝ : CommRing R
      p : Nat
      hp : Nat.Prime p
      I : Ideal R
      hI_max : I.IsMaximal
      h_charP : CharP (HasQuotient.Quotient R I) p
      ⊢ Exists fun I => And (Ne I Top.top) (CharP (HasQuotient.Quotient R I) p)
    -/
    use I
    /-
      case h
      R : Type u_1
      inst✝ : CommRing R
      p : Nat
      hp : Nat.Prime p
      I : Ideal R
      hI_max : I.IsMaximal
      h_charP : CharP (HasQuotient.Quotient R I) p
      ⊢ And (Ne I Top.top) (CharP (HasQuotient.Quotient R I) p)
    -/
    exact ⟨Ideal.IsMaximal.ne_top hI_max, h_charP⟩
    /-
      🎉 no goals
    -/


/-- `ℚ`-algebra implies equal characteristic. -/
theorem of_algebraRat [Algebra ℚ R] : ∀ I : Ideal R, I ≠ ⊤ → CharZero (R ⧸ I) := by
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : Algebra Rat R
    ⊢ ∀ (I : Ideal R), Ne I Top.top → CharZero (HasQuotient.Quotient R I)
  -/
  intro I hI
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : Algebra Rat R
    I : Ideal R
    hI : Ne I Top.top
    ⊢ CharZero (HasQuotient.Quotient R I)
  -/
  constructor
  /-
    case cast_injective
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : Algebra Rat R
    I : Ideal R
    hI : Ne I Top.top
    ⊢ Function.Injective Nat.cast
  -/
  intro a b h_ab
  /-
    case cast_injective
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : Algebra Rat R
    I : Ideal R
    hI : Ne I Top.top
    a b : Nat
    h_ab : Eq ↑a ↑b
    ⊢ Eq a b
  -/
  contrapose! hI
  -- `↑a - ↑b` is a unit contained in `I`, which contradicts `I ≠ ⊤`.
  /-
    case cast_injective
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : Algebra Rat R
    I : Ideal R
    a b : Nat
    h_ab : Eq ↑a ↑b
    hI : Ne a b
    ⊢ Eq I Top.top
  -/
  refine I.eq_top_of_isUnit_mem ?_ (IsUnit.map (algebraMap ℚ R) (IsUnit.mk0 (a - b : ℚ) ?_))
    /-
      case cast_injective.refine_1
      R : Type u_1
      inst✝¹ : CommRing R
      inst✝ : Algebra Rat R
      I : Ideal R
      a b : Nat
      h_ab : Eq ↑a ↑b
      hI : Ne a b
      ⊢ Membership.mem I ((algebraMap Rat R) (HSub.hSub ↑a ↑b))
    -/
  · simpa only [← Ideal.Quotient.eq_zero_iff_mem, map_sub, sub_eq_zero, map_natCast]
    /-
      🎉 no goals
    -/
  /-
    case cast_injective.refine_2
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : Algebra Rat R
    I : Ideal R
    a b : Nat
    h_ab : Eq ↑a ↑b
    hI : Ne a b
    ⊢ Ne (HSub.hSub ↑a ↑b) 0
  -/
  simpa only [Ne, sub_eq_zero] using (@Nat.cast_injective ℚ _ _).ne hI
  /-
    🎉 no goals
  -/


/-- Internal: Not intended to be used outside this local construction. -/
theorem PNat.isUnit_natCast [h : Fact (∀ I : Ideal R, I ≠ ⊤ → CharZero (R ⧸ I))]
    (n : ℕ+) : IsUnit (n : R) := by
  -- `n : R` is a unit iff `(n)` is not a proper ideal in `R`.
  /-
    R : Type u_1
    inst✝ : CommRing R
    h : Fact (∀ (I : Ideal R), Ne I Top.top → CharZero (HasQuotient.Quotient R I))
    n : PNat
    ⊢ IsUnit ↑↑n
  -/
  rw [← Ideal.span_singleton_eq_top]
  -- So by contrapositive, we should show the quotient does not have characteristic zero.
  /-
    R : Type u_1
    inst✝ : CommRing R
    h : Fact (∀ (I : Ideal R), Ne I Top.top → CharZero (HasQuotient.Quotient R I))
    n : PNat
    ⊢ Eq (Ideal.span (Singleton.singleton ↑↑n)) Top.top
  -/
  apply not_imp_comm.mp (h.elim (Ideal.span {↑n}))
  /-
    R : Type u_1
    inst✝ : CommRing R
    h : Fact (∀ (I : Ideal R), Ne I Top.top → CharZero (HasQuotient.Quotient R I))
    n : PNat
    ⊢ Not (CharZero (HasQuotient.Quotient R (Ideal.span (Singleton.singleton ↑↑n))))
  -/
  intro h_char_zero
  -- In particular, the image of `n` in the quotient should be nonzero.
  /-
    R : Type u_1
    inst✝ : CommRing R
    h : Fact (∀ (I : Ideal R), Ne I Top.top → CharZero (HasQuotient.Quotient R I))
    n : PNat
    h_char_zero : CharZero (HasQuotient.Quotient R (Ideal.span (Singleton.singleto …
    ⊢ False
  -/
  apply h_char_zero.cast_injective.ne n.ne_zero
  -- But `n` generates the ideal, so its image is clearly zero.
  /-
    R : Type u_1
    inst✝ : CommRing R
    h : Fact (∀ (I : Ideal R), Ne I Top.top → CharZero (HasQuotient.Quotient R I))
    n : PNat
    h_char_zero : CharZero (HasQuotient.Quotient R (Ideal.span (Singleton.singleto …
    ⊢ Eq ↑↑n ↑0
  -/
  rw [← map_natCast (Ideal.Quotient.mk _), Nat.cast_zero, Ideal.Quotient.eq_zero_iff_mem]
  /-
    R : Type u_1
    inst✝ : CommRing R
    h : Fact (∀ (I : Ideal R), Ne I Top.top → CharZero (HasQuotient.Quotient R I))
    n : PNat
    h_char_zero : CharZero (HasQuotient.Quotient R (Ideal.span (Singleton.singleto …
    ⊢ Membership.mem (Ideal.span (Singleton.singleton ↑↑n)) ↑↑n
  -/
  exact Ideal.subset_span (Set.mem_singleton _)
  /-
    🎉 no goals
  -/


@[coe]
noncomputable def pnatCast [Fact (∀ I : Ideal R, I ≠ ⊤ → CharZero (R ⧸ I))] : ℕ+ → Rˣ :=
  fun n => (PNat.isUnit_natCast n).unit


/-- Internal: Not intended to be used outside this local construction. -/
noncomputable instance coePNatUnits
    [Fact (∀ I : Ideal R, I ≠ ⊤ → CharZero (R ⧸ I))] : Coe ℕ+ Rˣ :=
  ⟨EqualCharZero.pnatCast⟩


/-- Internal: Not intended to be used outside this local construction. -/
@[simp]
theorem pnatCast_one [Fact (∀ I : Ideal R, I ≠ ⊤ → CharZero (R ⧸ I))] : ((1 : ℕ+) : Rˣ) = 1 := by
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : Fact (∀ (I : Ideal R), Ne I Top.top → CharZero (HasQuotient.Quotient R …
    ⊢ Eq (↑1) 1
  -/
  apply Units.ext
  /-
    case a
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : Fact (∀ (I : Ideal R), Ne I Top.top → CharZero (HasQuotient.Quotient R …
    ⊢ Eq ↑↑1 ↑1
  -/
  rw [Units.val_one]
  /-
    case a
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : Fact (∀ (I : Ideal R), Ne I Top.top → CharZero (HasQuotient.Quotient R …
    ⊢ Eq (↑↑1) 1
  -/
  change ((PNat.isUnit_natCast (R := R) 1).unit : R) = 1
  /-
    case a
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : Fact (∀ (I : Ideal R), Ne I Top.top → CharZero (HasQuotient.Quotient R …
    ⊢ Eq (↑⋯.unit) 1
  -/
  rw [IsUnit.unit_spec (PNat.isUnit_natCast 1)]
  /-
    case a
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : Fact (∀ (I : Ideal R), Ne I Top.top → CharZero (HasQuotient.Quotient R …
    ⊢ Eq (↑↑1) 1
  -/
  rw [PNat.one_coe, Nat.cast_one]
  /-
    🎉 no goals
  -/


/-- Internal: Not intended to be used outside this local construction. -/
@[simp]
theorem pnatCast_eq_natCast [Fact (∀ I : Ideal R, I ≠ ⊤ → CharZero (R ⧸ I))] (n : ℕ+) :
    ((n : Rˣ) : R) = ↑n := by
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : Fact (∀ (I : Ideal R), Ne I Top.top → CharZero (HasQuotient.Quotient R …
    n : PNat
    ⊢ Eq ↑↑n ↑↑n
  -/
  change ((PNat.isUnit_natCast (R := R) n).unit : R) = ↑n
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : Fact (∀ (I : Ideal R), Ne I Top.top → CharZero (HasQuotient.Quotient R …
    n : PNat
    ⊢ Eq ↑⋯.unit ↑↑n
  -/
  simp only [IsUnit.unit_spec]
  /-
    🎉 no goals
  -/


/-- Equal characteristic implies `ℚ`-algebra. -/
noncomputable def algebraRat (h : ∀ I : Ideal R, I ≠ ⊤ → CharZero (R ⧸ I)) :
    Algebra ℚ R :=
  haveI : Fact (∀ I : Ideal R, I ≠ ⊤ → CharZero (R ⧸ I)) := ⟨h⟩
  RingHom.toAlgebra
  { toFun := fun x => x.num /ₚ ↑x.pnatDen
                    /-
                      R : Type u_1
                      inst✝ : CommRing R
                      h : ∀ (I : Ideal R), Ne I Top.top → CharZero (HasQuotient.Quotient R I)
                      this : Fact (∀ (I : Ideal R), Ne I Top.top → CharZero (HasQuotient.Quotient R  …
                      ⊢ Eq ((↑{ toFun := fun x => divp ↑x.num ↑x.pnatDen, map_one' := ⋯, map_mul' := …
                    -/
                   /-
                     R : Type u_1
                     inst✝ : CommRing R
                     h : ∀ (I : Ideal R), Ne I Top.top → CharZero (HasQuotient.Quotient R I)
                     this : Fact (∀ (I : Ideal R), Ne I Top.top → CharZero (HasQuotient.Quotient R  …
                     ⊢ Eq ((fun x => divp ↑x.num ↑x.pnatDen) 1) 1
                   -/
    map_zero' := by simp [divp]
                   /-
                     🎉 no goals
                   -/
                    /-
                      🎉 no goals
                    -/
      /-
        R : Type u_1
        inst✝ : CommRing R
        h : ∀ (I : Ideal R), Ne I Top.top → CharZero (HasQuotient.Quotient R I)
        this : Fact (∀ (I : Ideal R), Ne I Top.top → CharZero (HasQuotient.Quotient R  …
        ⊢ ∀ (x y : Rat), Eq ({ toFun := fun x => divp ↑x.num ↑x.pnatDen, map_one' := ⋯ …
      -/
    map_one' := by simp
      /-
        R : Type u_1
        inst✝ : CommRing R
        h : ∀ (I : Ideal R), Ne I Top.top → CharZero (HasQuotient.Quotient R I)
        this : Fact (∀ (I : Ideal R), Ne I Top.top → CharZero (HasQuotient.Quotient R  …
        a b : Rat
        ⊢ Eq ({ toFun := fun x => divp ↑x.num ↑x.pnatDen, map_one' := ⋯ }.toFun (HMul. …
      -/
    map_mul' := by
      /-
        R : Type u_1
        inst✝ : CommRing R
        h : ∀ (I : Ideal R), Ne I Top.top → CharZero (HasQuotient.Quotient R I)
        this : Fact (∀ (I : Ideal R), Ne I Top.top → CharZero (HasQuotient.Quotient R  …
        a b : Rat
        ⊢ Eq (HMul.hMul (↑(HMul.hMul a b).num) (HMul.hMul ↑b.den ↑a.den)) (HMul.hMul ( …
      -/
      intro a b
        /-
          R : Type u_1
          inst✝ : CommRing R
          h : ∀ (I : Ideal R), Ne I Top.top → CharZero (HasQuotient.Quotient R I)
          this : Fact (∀ (I : Ideal R), Ne I Top.top → CharZero (HasQuotient.Quotient R  …
          a b : Rat
          ⊢ Eq (HMul.hMul (↑(HMul.hMul a b).num) (HMul.hMul ↑b.den ↑a.den)) ↑(HMul.hMul  …
        -/
      field_simp
        /-
          R : Type u_1
          inst✝ : CommRing R
          h : ∀ (I : Ideal R), Ne I Top.top → CharZero (HasQuotient.Quotient R I)
          this : Fact (∀ (I : Ideal R), Ne I Top.top → CharZero (HasQuotient.Quotient R  …
          a b : Rat
          ⊢ Eq (HMul.hMul (↑(HMul.hMul a b).num) (HMul.hMul ↑b.den ↑a.den)) (HMul.hMul ( …
        -/
      trans (↑((a * b).num * a.den * b.den) : R)
        /-
          🎉 no goals
        -/
      /-
        R : Type u_1
        inst✝ : CommRing R
        h : ∀ (I : Ideal R), Ne I Top.top → CharZero (HasQuotient.Quotient R I)
        this : Fact (∀ (I : Ideal R), Ne I Top.top → CharZero (HasQuotient.Quotient R  …
        a b : Rat
        ⊢ Eq (↑(HMul.hMul (HMul.hMul (HMul.hMul a b).num ↑a.den) ↑b.den)) (HMul.hMul ( …
      -/
      · simp_rw [Int.cast_mul, Int.cast_natCast]
      /-
        R : Type u_1
        inst✝ : CommRing R
        h : ∀ (I : Ideal R), Ne I Top.top → CharZero (HasQuotient.Quotient R I)
        this : Fact (∀ (I : Ideal R), Ne I Top.top → CharZero (HasQuotient.Quotient R  …
        a b : Rat
        ⊢ Eq (↑(HMul.hMul (HMul.hMul a.num b.num) ↑(HMul.hMul a b).den)) (HMul.hMul (H …
      -/
        ring
      /-
        🎉 no goals
      -/
      rw [Rat.mul_num_den' a b]
      simp
    map_add' := by
      /-
        R : Type u_1
        inst✝ : CommRing R
        h : ∀ (I : Ideal R), Ne I Top.top → CharZero (HasQuotient.Quotient R I)
        this : Fact (∀ (I : Ideal R), Ne I Top.top → CharZero (HasQuotient.Quotient R  …
        ⊢ ∀ (x y : Rat), Eq ((↑{ toFun := fun x => divp ↑x.num ↑x.pnatDen, map_one' := …
      -/
      intro a b
      /-
        R : Type u_1
        inst✝ : CommRing R
        h : ∀ (I : Ideal R), Ne I Top.top → CharZero (HasQuotient.Quotient R I)
        this : Fact (∀ (I : Ideal R), Ne I Top.top → CharZero (HasQuotient.Quotient R  …
        a b : Rat
        ⊢ Eq ((↑{ toFun := fun x => divp ↑x.num ↑x.pnatDen, map_one' := ⋯, map_mul' := …
      -/
      field_simp
      /-
        R : Type u_1
        inst✝ : CommRing R
        h : ∀ (I : Ideal R), Ne I Top.top → CharZero (HasQuotient.Quotient R I)
        this : Fact (∀ (I : Ideal R), Ne I Top.top → CharZero (HasQuotient.Quotient R  …
        a b : Rat
        ⊢ Eq (HMul.hMul (↑(HAdd.hAdd a b).num) (HMul.hMul ↑b.den ↑a.den)) (HMul.hMul ( …
      -/
      trans (↑((a + b).num * a.den * b.den) : R)
        /-
          R : Type u_1
          inst✝ : CommRing R
          h : ∀ (I : Ideal R), Ne I Top.top → CharZero (HasQuotient.Quotient R I)
          this : Fact (∀ (I : Ideal R), Ne I Top.top → CharZero (HasQuotient.Quotient R  …
          a b : Rat
          ⊢ Eq (HMul.hMul (↑(HAdd.hAdd a b).num) (HMul.hMul ↑b.den ↑a.den)) ↑(HMul.hMul  …
        -/
      · simp_rw [Int.cast_mul, Int.cast_natCast]
        /-
          R : Type u_1
          inst✝ : CommRing R
          h : ∀ (I : Ideal R), Ne I Top.top → CharZero (HasQuotient.Quotient R I)
          this : Fact (∀ (I : Ideal R), Ne I Top.top → CharZero (HasQuotient.Quotient R  …
          a b : Rat
          ⊢ Eq (HMul.hMul (↑(HAdd.hAdd a b).num) (HMul.hMul ↑b.den ↑a.den)) (HMul.hMul ( …
        -/
        ring
        /-
          🎉 no goals
        -/
      /-
        R : Type u_1
        inst✝ : CommRing R
        h : ∀ (I : Ideal R), Ne I Top.top → CharZero (HasQuotient.Quotient R I)
        this : Fact (∀ (I : Ideal R), Ne I Top.top → CharZero (HasQuotient.Quotient R  …
        a b : Rat
        ⊢ Eq (↑(HMul.hMul (HMul.hMul (HAdd.hAdd a b).num ↑a.den) ↑b.den)) (HMul.hMul ( …
      -/
      rw [Rat.add_num_den' a b]
      /-
        R : Type u_1
        inst✝ : CommRing R
        h : ∀ (I : Ideal R), Ne I Top.top → CharZero (HasQuotient.Quotient R I)
        this : Fact (∀ (I : Ideal R), Ne I Top.top → CharZero (HasQuotient.Quotient R  …
        a b : Rat
        ⊢ Eq (↑(HMul.hMul (HAdd.hAdd (HMul.hMul a.num ↑b.den) (HMul.hMul b.num ↑a.den) …
      -/
      simp }
      /-
        🎉 no goals
      -/


/-- Not mixed characteristic implies equal characteristic. -/
theorem of_not_mixedCharZero [CharZero R] (h : ∀ p > 0, ¬MixedCharZero R p) :
    ∀ I : Ideal R, I ≠ ⊤ → CharZero (R ⧸ I) := by
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : CharZero R
    h : ∀ (p : Nat), GT.gt p 0 → Not (MixedCharZero R p)
    ⊢ ∀ (I : Ideal R), Ne I Top.top → CharZero (HasQuotient.Quotient R I)
  -/
  intro I hI_ne_top
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : CharZero R
    h : ∀ (p : Nat), GT.gt p 0 → Not (MixedCharZero R p)
    I : Ideal R
    hI_ne_top : Ne I Top.top
    ⊢ CharZero (HasQuotient.Quotient R I)
  -/
  suffices CharP (R ⧸ I) 0 from CharP.charP_to_charZero _
  cases CharP.exists (R ⧸ I) with
  | intro p hp =>
    cases p with
    | zero => exact hp
    | succ p =>
      have h_mixed : MixedCharZero R p.succ := ⟨⟨I, ⟨hI_ne_top, hp⟩⟩⟩
      exact absurd h_mixed (h p.succ p.succ_pos)


/-- Equal characteristic implies not mixed characteristic. -/
theorem to_not_mixedCharZero (h : ∀ I : Ideal R, I ≠ ⊤ → CharZero (R ⧸ I)) :
    ∀ p > 0, ¬MixedCharZero R p := by
  /-
    R : Type u_1
    inst✝ : CommRing R
    h : ∀ (I : Ideal R), Ne I Top.top → CharZero (HasQuotient.Quotient R I)
    ⊢ ∀ (p : Nat), GT.gt p 0 → Not (MixedCharZero R p)
  -/
  intro p p_pos
  /-
    R : Type u_1
    inst✝ : CommRing R
    h : ∀ (I : Ideal R), Ne I Top.top → CharZero (HasQuotient.Quotient R I)
    p : Nat
    p_pos : GT.gt p 0
    ⊢ Not (MixedCharZero R p)
  -/
  by_contra hp_mixedChar
  /-
    R : Type u_1
    inst✝ : CommRing R
    h : ∀ (I : Ideal R), Ne I Top.top → CharZero (HasQuotient.Quotient R I)
    p : Nat
    p_pos : GT.gt p 0
    hp_mixedChar : MixedCharZero R p
    ⊢ False
  -/
  rcases hp_mixedChar.charP_quotient with ⟨I, hI_ne_top, hI_p⟩
  /-
    case intro.intro
    R : Type u_1
    inst✝ : CommRing R
    h : ∀ (I : Ideal R), Ne I Top.top → CharZero (HasQuotient.Quotient R I)
    p : Nat
    p_pos : GT.gt p 0
    hp_mixedChar : MixedCharZero R p
    I : Ideal R
    hI_ne_top : Ne I Top.top
    hI_p : CharP (HasQuotient.Quotient R I) p
    ⊢ False
  -/
  replace hI_zero : CharP (R ⧸ I) 0 := @CharP.ofCharZero _ _ (h I hI_ne_top)
  /-
    case intro.intro
    R : Type u_1
    inst✝ : CommRing R
    h : ∀ (I : Ideal R), Ne I Top.top → CharZero (HasQuotient.Quotient R I)
    p : Nat
    p_pos : GT.gt p 0
    hp_mixedChar : MixedCharZero R p
    I : Ideal R
    hI_ne_top : Ne I Top.top
    hI_p : CharP (HasQuotient.Quotient R I) p
    hI_zero : CharP (HasQuotient.Quotient R I) 0
    ⊢ False
  -/
  exact absurd (CharP.eq (R ⧸ I) hI_p hI_zero) (ne_of_gt p_pos)
  /-
    🎉 no goals
  -/


/--
A ring of characteristic zero has equal characteristic iff it does not
have mixed characteristic for any `p`.
-/
theorem iff_not_mixedCharZero [CharZero R] :
    (∀ I : Ideal R, I ≠ ⊤ → CharZero (R ⧸ I)) ↔ ∀ p > 0, ¬MixedCharZero R p :=
  ⟨to_not_mixedCharZero R, of_not_mixedCharZero R⟩


/-- A ring is a `ℚ`-algebra iff it has equal characteristic zero. -/
theorem nonempty_algebraRat_iff :
    Nonempty (Algebra ℚ R) ↔ ∀ I : Ideal R, I ≠ ⊤ → CharZero (R ⧸ I) := by
  /-
    R : Type u_1
    inst✝ : CommRing R
    ⊢ Iff (Nonempty (Algebra Rat R)) (∀ (I : Ideal R), Ne I Top.top → CharZero (Ha …
  -/
  constructor
    /-
      case mp
      R : Type u_1
      inst✝ : CommRing R
      ⊢ Nonempty (Algebra Rat R) → ∀ (I : Ideal R), Ne I Top.top → CharZero (HasQuot …
    -/
  · intro h_alg
    /-
      case mp
      R : Type u_1
      inst✝ : CommRing R
      h_alg : Nonempty (Algebra Rat R)
      ⊢ ∀ (I : Ideal R), Ne I Top.top → CharZero (HasQuotient.Quotient R I)
    -/
    haveI h_alg' : Algebra ℚ R := h_alg.some
    /-
      case mp
      R : Type u_1
      inst✝ : CommRing R
      h_alg : Nonempty (Algebra Rat R)
      h_alg' : Algebra Rat R
      ⊢ ∀ (I : Ideal R), Ne I Top.top → CharZero (HasQuotient.Quotient R I)
    -/
    apply of_algebraRat
    /-
      🎉 no goals
    -/
    /-
      case mpr
      R : Type u_1
      inst✝ : CommRing R
      ⊢ (∀ (I : Ideal R), Ne I Top.top → CharZero (HasQuotient.Quotient R I)) → None …
    -/
  · intro h
    /-
      case mpr
      R : Type u_1
      inst✝ : CommRing R
      h : ∀ (I : Ideal R), Ne I Top.top → CharZero (HasQuotient.Quotient R I)
      ⊢ Nonempty (Algebra Rat R)
    -/
    apply Nonempty.intro
    /-
      case mpr.val
      R : Type u_1
      inst✝ : CommRing R
      h : ∀ (I : Ideal R), Ne I Top.top → CharZero (HasQuotient.Quotient R I)
      ⊢ Algebra Rat R
    -/
    exact algebraRat h
    /-
      🎉 no goals
    -/


/--
A ring of characteristic zero is not a `ℚ`-algebra iff it has mixed characteristic for some `p`.
-/
theorem isEmpty_algebraRat_iff_mixedCharZero [CharZero R] :
    IsEmpty (Algebra ℚ R) ↔ ∃ p > 0, MixedCharZero R p := by
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : CharZero R
    ⊢ Iff (IsEmpty (Algebra Rat R)) (Exists fun p => And (GT.gt p 0) (MixedCharZer …
  -/
  rw [← not_iff_not]
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : CharZero R
    ⊢ Iff (Not (IsEmpty (Algebra Rat R))) (Not (Exists fun p => And (GT.gt p 0) (M …
  -/
  push_neg
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : CharZero R
    ⊢ Iff (Not (IsEmpty (Algebra Rat R))) (∀ (p : Nat), GT.gt p 0 → Not (MixedChar …
  -/
  rw [not_isEmpty_iff, ← EqualCharZero.iff_not_mixedCharZero]
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : CharZero R
    ⊢ Iff (Nonempty (Algebra Rat R)) (∀ (I : Ideal R), Ne I Top.top → CharZero (Ha …
  -/
  apply EqualCharZero.nonempty_algebraRat_iff
  /-
    🎉 no goals
  -/


/-- Split a `Prop` in characteristic zero into equal and mixed characteristic. -/
theorem split_equalCharZero_mixedCharZero [CharZero R] (h_equal : Algebra ℚ R → P)
    (h_mixed : ∀ p : ℕ, Nat.Prime p → MixedCharZero R p → P) : P := by
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    P : Prop
    inst✝ : CharZero R
    h_equal : Algebra Rat R → P
    h_mixed : ∀ (p : Nat), Nat.Prime p → MixedCharZero R p → P
    ⊢ P
  -/
  by_cases h : ∃ p > 0, MixedCharZero R p
    /-
      case pos
      R : Type u_1
      inst✝¹ : CommRing R
      P : Prop
      inst✝ : CharZero R
      h_equal : Algebra Rat R → P
      h_mixed : ∀ (p : Nat), Nat.Prime p → MixedCharZero R p → P
      h : Exists fun p => And (GT.gt p 0) (MixedCharZero R p)
      ⊢ P
    -/
  · rcases h with ⟨p, ⟨H, hp⟩⟩
    /-
      case pos.intro.intro
      R : Type u_1
      inst✝¹ : CommRing R
      P : Prop
      inst✝ : CharZero R
      h_equal : Algebra Rat R → P
      h_mixed : ∀ (p : Nat), Nat.Prime p → MixedCharZero R p → P
      p : Nat
      H : GT.gt p 0
      hp : MixedCharZero R p
      ⊢ P
    -/
    rw [← MixedCharZero.reduce_to_p_prime] at h_mixed
    /-
      case pos.intro.intro
      R : Type u_1
      inst✝¹ : CommRing R
      P : Prop
      inst✝ : CharZero R
      h_equal : Algebra Rat R → P
      h_mixed : ∀ (p : Nat), GT.gt p 0 → MixedCharZero R p → P
      p : Nat
      H : GT.gt p 0
      hp : MixedCharZero R p
      ⊢ P
    -/
    exact h_mixed p H hp
    /-
      🎉 no goals
    -/
    /-
      case neg
      R : Type u_1
      inst✝¹ : CommRing R
      P : Prop
      inst✝ : CharZero R
      h_equal : Algebra Rat R → P
      h_mixed : ∀ (p : Nat), Nat.Prime p → MixedCharZero R p → P
      h : Not (Exists fun p => And (GT.gt p 0) (MixedCharZero R p))
      ⊢ P
    -/
  · apply h_equal
    /-
      case neg
      R : Type u_1
      inst✝¹ : CommRing R
      P : Prop
      inst✝ : CharZero R
      h_equal : Algebra Rat R → P
      h_mixed : ∀ (p : Nat), Nat.Prime p → MixedCharZero R p → P
      h : Not (Exists fun p => And (GT.gt p 0) (MixedCharZero R p))
      ⊢ Algebra Rat R
    -/
    rw [← isEmpty_algebraRat_iff_mixedCharZero, not_isEmpty_iff] at h
    /-
      case neg
      R : Type u_1
      inst✝¹ : CommRing R
      P : Prop
      inst✝ : CharZero R
      h_equal : Algebra Rat R → P
      h_mixed : ∀ (p : Nat), Nat.Prime p → MixedCharZero R p → P
      h : Nonempty (Algebra Rat R)
      ⊢ Algebra Rat R
    -/
    exact h.some
    /-
      🎉 no goals
    -/


/--
Split any `Prop` over `R` into the three cases:
- positive characteristic.
- equal characteristic zero.
- mixed characteristic `(0, p)`.
-/
theorem split_by_characteristic (h_pos : ∀ p : ℕ, p ≠ 0 → CharP R p → P) (h_equal : Algebra ℚ R → P)
    (h_mixed : ∀ p : ℕ, Nat.Prime p → MixedCharZero R p → P) : P := by
  cases CharP.exists R with
  | intro p p_charP =>
    by_cases h : p = 0
    · rw [h] at p_charP
      haveI h0 : CharZero R := CharP.charP_to_charZero R
      exact split_equalCharZero_mixedCharZero R h_equal h_mixed
    · exact h_pos p h p_charP


/--
In an `IsDomain R`, split any `Prop` over `R` into the three cases:
- *prime* characteristic.
- equal characteristic zero.
- mixed characteristic `(0, p)`.
-/
theorem split_by_characteristic_domain [IsDomain R] (h_pos : ∀ p : ℕ, Nat.Prime p → CharP R p → P)
    (h_equal : Algebra ℚ R → P) (h_mixed : ∀ p : ℕ, Nat.Prime p → MixedCharZero R p → P) : P := by
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    P : Prop
    inst✝ : IsDomain R
    h_pos : ∀ (p : Nat), Nat.Prime p → CharP R p → P
    h_equal : Algebra Rat R → P
    h_mixed : ∀ (p : Nat), Nat.Prime p → MixedCharZero R p → P
    ⊢ P
  -/
  refine split_by_characteristic R ?_ h_equal h_mixed
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    P : Prop
    inst✝ : IsDomain R
    h_pos : ∀ (p : Nat), Nat.Prime p → CharP R p → P
    h_equal : Algebra Rat R → P
    h_mixed : ∀ (p : Nat), Nat.Prime p → MixedCharZero R p → P
    ⊢ ∀ (p : Nat), Ne p 0 → CharP R p → P
  -/
  intro p p_pos p_char
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    P : Prop
    inst✝ : IsDomain R
    h_pos : ∀ (p : Nat), Nat.Prime p → CharP R p → P
    h_equal : Algebra Rat R → P
    h_mixed : ∀ (p : Nat), Nat.Prime p → MixedCharZero R p → P
    p : Nat
    p_pos : Ne p 0
    p_char : CharP R p
    ⊢ P
  -/
  have p_prime : Nat.Prime p := or_iff_not_imp_right.mp (CharP.char_is_prime_or_zero R p) p_pos
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    P : Prop
    inst✝ : IsDomain R
    h_pos : ∀ (p : Nat), Nat.Prime p → CharP R p → P
    h_equal : Algebra Rat R → P
    h_mixed : ∀ (p : Nat), Nat.Prime p → MixedCharZero R p → P
    p : Nat
    p_pos : Ne p 0
    p_char : CharP R p
    p_prime : Nat.Prime p
    ⊢ P
  -/
  exact h_pos p p_prime p_char
  /-
    🎉 no goals
  -/


/--
In a local ring `R`, split any predicate over `R` into the three cases:
- *prime power* characteristic.
- equal characteristic zero.
- mixed characteristic `(0, p)`.
-/
theorem split_by_characteristic_localRing [IsLocalRing R]
    (h_pos : ∀ p : ℕ, IsPrimePow p → CharP R p → P) (h_equal : Algebra ℚ R → P)
    (h_mixed : ∀ p : ℕ, Nat.Prime p → MixedCharZero R p → P) : P := by
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    P : Prop
    inst✝ : IsLocalRing R
    h_pos : ∀ (p : Nat), IsPrimePow p → CharP R p → P
    h_equal : Algebra Rat R → P
    h_mixed : ∀ (p : Nat), Nat.Prime p → MixedCharZero R p → P
    ⊢ P
  -/
  refine split_by_characteristic R ?_ h_equal h_mixed
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    P : Prop
    inst✝ : IsLocalRing R
    h_pos : ∀ (p : Nat), IsPrimePow p → CharP R p → P
    h_equal : Algebra Rat R → P
    h_mixed : ∀ (p : Nat), Nat.Prime p → MixedCharZero R p → P
    ⊢ ∀ (p : Nat), Ne p 0 → CharP R p → P
  -/
  intro p p_pos p_char
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    P : Prop
    inst✝ : IsLocalRing R
    h_pos : ∀ (p : Nat), IsPrimePow p → CharP R p → P
    h_equal : Algebra Rat R → P
    h_mixed : ∀ (p : Nat), Nat.Prime p → MixedCharZero R p → P
    p : Nat
    p_pos : Ne p 0
    p_char : CharP R p
    ⊢ P
  -/
  have p_ppow : IsPrimePow (p : ℕ) := or_iff_not_imp_left.mp (charP_zero_or_prime_power R p) p_pos
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    P : Prop
    inst✝ : IsLocalRing R
    h_pos : ∀ (p : Nat), IsPrimePow p → CharP R p → P
    h_equal : Algebra Rat R → P
    h_mixed : ∀ (p : Nat), Nat.Prime p → MixedCharZero R p → P
    p : Nat
    p_pos : Ne p 0
    p_char : CharP R p
    p_ppow : IsPrimePow p
    ⊢ P
  -/
  exact h_pos p p_ppow p_char
  /-
    🎉 no goals
  -/


