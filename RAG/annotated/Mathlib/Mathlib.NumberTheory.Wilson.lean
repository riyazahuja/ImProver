/-- **Wilson's Lemma**: the product of `1`, ..., `p-1` is `-1` modulo `p`. -/
@[simp]
theorem wilsons_lemma : ((p - 1)! : ZMod p) = -1 := by
  refine
    calc
      ((p - 1)! : ZMod p) = ∏ x ∈ Ico 1 (succ (p - 1)), (x : ZMod p) := by
        rw [← Finset.prod_Ico_id_eq_factorial, prod_natCast]
      _ = ∏ x : (ZMod p)ˣ, (x : ZMod p) := ?_
      _ = -1 := by
        -- Porting note: `simp` is less powerful.
        -- simp_rw [← Units.coeHom_apply, ← (Units.coeHom (ZMod p)).map_prod,
        --   prod_univ_units_id_eq_neg_one, Units.coeHom_apply, Units.val_neg, Units.val_one]
        simp_rw [← Units.coeHom_apply]
        rw [← map_prod (Units.coeHom (ZMod p))]
        simp_rw [prod_univ_units_id_eq_neg_one, Units.coeHom_apply, Units.val_neg, Units.val_one]
  /-
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    ⊢ Eq ((Finset.Ico 1 (HSub.hSub p 1).succ).prod fun x => ↑x) (Finset.univ.prod  …
  -/
  have hp : 0 < p := (Fact.out (p := p.Prime)).pos
  /-
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    hp : LT.lt 0 p
    ⊢ Eq ((Finset.Ico 1 (HSub.hSub p 1).succ).prod fun x => ↑x) (Finset.univ.prod  …
  -/
  symm
  /-
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    hp : LT.lt 0 p
    ⊢ Eq (Finset.univ.prod fun x => ↑x) ((Finset.Ico 1 (HSub.hSub p 1).succ).prod  …
  -/
  refine prod_bij (fun a _ => (a : ZMod p).val) ?_ ?_ ?_ ?_
    /-
      case refine_1
      p : Nat
      inst✝ : Fact (Nat.Prime p)
      hp : LT.lt 0 p
      ⊢ ∀ (a : Units (ZMod p)) (ha : Membership.mem Finset.univ a), Membership.mem ( …
    -/
  · intro a ha
    /-
      case refine_1
      p : Nat
      inst✝ : Fact (Nat.Prime p)
      hp : LT.lt 0 p
      a : Units (ZMod p)
      ha : Membership.mem Finset.univ a
      ⊢ Membership.mem (Finset.Ico 1 (HSub.hSub p 1).succ) ((fun a x => (↑a).val) a  …
    -/
    rw [mem_Ico, ← Nat.succ_sub hp, Nat.add_one_sub_one]
    /-
      case refine_1
      p : Nat
      inst✝ : Fact (Nat.Prime p)
      hp : LT.lt 0 p
      a : Units (ZMod p)
      ha : Membership.mem Finset.univ a
      ⊢ And (LE.le 1 ((fun a x => (↑a).val) a ha)) (LT.lt ((fun a x => (↑a).val) a h …
    -/
    constructor
      /-
        case refine_1.left
        p : Nat
        inst✝ : Fact (Nat.Prime p)
        hp : LT.lt 0 p
        a : Units (ZMod p)
        ha : Membership.mem Finset.univ a
        ⊢ LE.le 1 ((fun a x => (↑a).val) a ha)
      -/
    · apply Nat.pos_of_ne_zero; rw [← @val_zero p]
      /-
        case refine_1.left.a
        p : Nat
        inst✝ : Fact (Nat.Prime p)
        hp : LT.lt 0 p
        a : Units (ZMod p)
        ha : Membership.mem Finset.univ a
        ⊢ Ne ((fun a x => (↑a).val) a ha) (ZMod.val 0)
      -/
      intro h; apply Units.ne_zero a (val_injective p h)
               /-
                 🎉 no goals
               -/
      /-
        case refine_1.right
        p : Nat
        inst✝ : Fact (Nat.Prime p)
        hp : LT.lt 0 p
        a : Units (ZMod p)
        ha : Membership.mem Finset.univ a
        ⊢ LT.lt ((fun a x => (↑a).val) a ha) p
      -/
    · exact val_lt _
      /-
        🎉 no goals
      -/
    /-
      case refine_2
      p : Nat
      inst✝ : Fact (Nat.Prime p)
      hp : LT.lt 0 p
      ⊢ ∀ (a₁ : Units (ZMod p)) (ha₁ : Membership.mem Finset.univ a₁) (a₂ : Units (Z …
    -/
  · intro _ _ _ _ h; rw [Units.ext_iff]; exact val_injective p h
                                         /-
                                           🎉 no goals
                                         -/
    /-
      case refine_3
      p : Nat
      inst✝ : Fact (Nat.Prime p)
      hp : LT.lt 0 p
      ⊢ ∀ (b : Nat), Membership.mem (Finset.Ico 1 (HSub.hSub p 1).succ) b → Exists f …
    -/
  · intro b hb
    /-
      case refine_3
      p : Nat
      inst✝ : Fact (Nat.Prime p)
      hp : LT.lt 0 p
      b : Nat
      hb : Membership.mem (Finset.Ico 1 (HSub.hSub p 1).succ) b
      ⊢ Exists fun a => Exists fun ha => Eq ((fun a x => (↑a).val) a ha) b
    -/
    rw [mem_Ico, Nat.succ_le_iff, ← succ_sub hp, Nat.add_one_sub_one, pos_iff_ne_zero] at hb
    /-
      case refine_3
      p : Nat
      inst✝ : Fact (Nat.Prime p)
      hp : LT.lt 0 p
      b : Nat
      hb : And (Ne b 0) (LT.lt b p)
      ⊢ Exists fun a => Exists fun ha => Eq ((fun a x => (↑a).val) a ha) b
    -/
    refine ⟨Units.mk0 b ?_, Finset.mem_univ _, ?_⟩
      /-
        case refine_3.refine_1
        p : Nat
        inst✝ : Fact (Nat.Prime p)
        hp : LT.lt 0 p
        b : Nat
        hb : And (Ne b 0) (LT.lt b p)
        ⊢ Ne (↑b) 0
      -/
    · intro h; apply hb.1; apply_fun val at h
      /-
        case refine_3.refine_1
        p : Nat
        inst✝ : Fact (Nat.Prime p)
        hp : LT.lt 0 p
        b : Nat
        hb : And (Ne b 0) (LT.lt b p)
        h : Eq (↑b).val (ZMod.val 0)
        ⊢ Eq b 0
      -/
      simpa only [val_cast_of_lt hb.right, val_zero] using h
      /-
        🎉 no goals
      -/
      /-
        case refine_3.refine_2
        p : Nat
        inst✝ : Fact (Nat.Prime p)
        hp : LT.lt 0 p
        b : Nat
        hb : And (Ne b 0) (LT.lt b p)
        ⊢ Eq ((fun a x => (↑a).val) (Units.mk0 ↑b ⋯) ⋯) b
      -/
    · simp only [val_cast_of_lt hb.right, Units.val_mk0]
      /-
        🎉 no goals
      -/
    /-
      case refine_4
      p : Nat
      inst✝ : Fact (Nat.Prime p)
      hp : LT.lt 0 p
      ⊢ ∀ (a : Units (ZMod p)) (ha : Membership.mem Finset.univ a), Eq ↑a ↑((fun a x …
    -/
  · rintro a -; simp only [cast_id, natCast_val]
                /-
                  🎉 no goals
                -/


@[simp]
theorem prod_Ico_one_prime : ∏ x ∈ Ico 1 p, (x : ZMod p) = -1 := by
  -- Porting note: was `conv in Ico 1 p =>`
  conv =>
    congr
    congr
    rw [← Nat.add_one_sub_one p, succ_sub (Fact.out (p := p.Prime)).pos]
  /-
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    ⊢ Eq ((Finset.Ico 1 (HSub.hSub p (Nat.succ 0)).succ).prod fun x => ↑x) (-1)
  -/
  rw [← prod_natCast, Finset.prod_Ico_id_eq_factorial, wilsons_lemma]
  /-
    🎉 no goals
  -/


/-- For `n ≠ 1`, `(n-1)!` is congruent to `-1` modulo `n` only if n is prime. -/
theorem prime_of_fac_equiv_neg_one (h : ((n - 1)! : ZMod n) = -1) (h1 : n ≠ 1) : Prime n := by
  /-
    n : Nat
    h : Eq (↑(HSub.hSub n 1).factorial) (-1)
    h1 : Ne n 1
    ⊢ Nat.Prime n
  -/
  rcases eq_or_ne n 0 with (rfl | h0)
    /-
      case inl
      h : Eq (↑(HSub.hSub 0 1).factorial) (-1)
      h1 : Ne 0 1
      ⊢ Nat.Prime 0
    -/
  · norm_num at h
    /-
      🎉 no goals
    -/
  /-
    case inr
    n : Nat
    h : Eq (↑(HSub.hSub n 1).factorial) (-1)
    h1 : Ne n 1
    h0 : Ne n 0
    ⊢ Nat.Prime n
  -/
  replace h1 : 1 < n := n.two_le_iff.mpr ⟨h0, h1⟩
  /-
    case inr
    n : Nat
    h : Eq (↑(HSub.hSub n 1).factorial) (-1)
    h0 : Ne n 0
    h1 : LT.lt 1 n
    ⊢ Nat.Prime n
  -/
  by_contra h2
  /-
    case inr
    n : Nat
    h : Eq (↑(HSub.hSub n 1).factorial) (-1)
    h0 : Ne n 0
    h1 : LT.lt 1 n
    h2 : Not (Nat.Prime n)
    ⊢ False
  -/
  obtain ⟨m, hm1, hm2 : 1 < m, hm3⟩ := exists_dvd_of_not_prime2 h1 h2
  /-
    case inr.intro.intro.intro
    n : Nat
    h : Eq (↑(HSub.hSub n 1).factorial) (-1)
    h0 : Ne n 0
    h1 : LT.lt 1 n
    h2 : Not (Nat.Prime n)
    m : Nat
    hm1 : Dvd.dvd m n
    hm2 : LT.lt 1 m
    hm3 : LT.lt m n
    ⊢ False
  -/
  have hm : m ∣ (n - 1)! := Nat.dvd_factorial (pos_of_gt hm2) (le_pred_of_lt hm3)
  /-
    case inr.intro.intro.intro
    n : Nat
    h : Eq (↑(HSub.hSub n 1).factorial) (-1)
    h0 : Ne n 0
    h1 : LT.lt 1 n
    h2 : Not (Nat.Prime n)
    m : Nat
    hm1 : Dvd.dvd m n
    hm2 : LT.lt 1 m
    hm3 : LT.lt m n
    hm : Dvd.dvd m (HSub.hSub n 1).factorial
    ⊢ False
  -/
  refine hm2.ne' (Nat.dvd_one.mp ((Nat.dvd_add_right hm).mp (hm1.trans ?_)))
  /-
    case inr.intro.intro.intro
    n : Nat
    h : Eq (↑(HSub.hSub n 1).factorial) (-1)
    h0 : Ne n 0
    h1 : LT.lt 1 n
    h2 : Not (Nat.Prime n)
    m : Nat
    hm1 : Dvd.dvd m n
    hm2 : LT.lt 1 m
    hm3 : LT.lt m n
    hm : Dvd.dvd m (HSub.hSub n 1).factorial
    ⊢ Dvd.dvd n (HAdd.hAdd (HSub.hSub n 1).factorial 1)
  -/
  rw [← ZMod.natCast_zmod_eq_zero_iff_dvd, cast_add, cast_one, h, neg_add_cancel]
  /-
    🎉 no goals
  -/


/-- **Wilson's Theorem**: For `n ≠ 1`, `(n-1)!` is congruent to `-1` modulo `n` iff n is prime. -/
theorem prime_iff_fac_equiv_neg_one (h : n ≠ 1) : Prime n ↔ ((n - 1)! : ZMod n) = -1 := by
  /-
    n : Nat
    h : Ne n 1
    ⊢ Iff (Nat.Prime n) (Eq (↑(HSub.hSub n 1).factorial) (-1))
  -/
  refine ⟨fun h1 => ?_, fun h2 => prime_of_fac_equiv_neg_one h2 h⟩
  /-
    n : Nat
    h : Ne n 1
    h1 : Nat.Prime n
    ⊢ Eq (↑(HSub.hSub n 1).factorial) (-1)
  -/
  haveI := Fact.mk h1
  /-
    n : Nat
    h : Ne n 1
    h1 : Nat.Prime n
    this : Fact (Nat.Prime n)
    ⊢ Eq (↑(HSub.hSub n 1).factorial) (-1)
  -/
  exact ZMod.wilsons_lemma n
  /-
    🎉 no goals
  -/


