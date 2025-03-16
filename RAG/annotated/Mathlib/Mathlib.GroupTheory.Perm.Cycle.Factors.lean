/-- `f.cycleOf x` is the cycle of the permutation `f` to which `x` belongs. -/
def cycleOf (f : Perm α) [DecidableRel f.SameCycle] (x : α) : Perm α :=
  ofSubtype (subtypePerm f fun _ => sameCycle_apply_right.symm : Perm { y // SameCycle f x y })


theorem cycleOf_apply (f : Perm α) [DecidableRel f.SameCycle] (x y : α) :
    cycleOf f x y = if SameCycle f x y then f y else y := by
  /-
    α : Type u_2
    f : Equiv.Perm α
    inst✝ : DecidableRel f.SameCycle
    x y : α
    ⊢ Eq ((f.cycleOf x) y) (ite (f.SameCycle x y) (f y) y)
  -/
  dsimp only [cycleOf]
  /-
    α : Type u_2
    f : Equiv.Perm α
    inst✝ : DecidableRel f.SameCycle
    x y : α
    ⊢ Eq ((Equiv.Perm.ofSubtype (f.subtypePerm ⋯)) y) (ite (f.SameCycle x y) (f y) …
  -/
  split_ifs with h
    /-
      case pos
      α : Type u_2
      f : Equiv.Perm α
      inst✝ : DecidableRel f.SameCycle
      x y : α
      h : f.SameCycle x y
      ⊢ Eq ((Equiv.Perm.ofSubtype (f.subtypePerm ⋯)) y) (f y)
    -/
  · apply ofSubtype_apply_of_mem
    /-
      case pos.ha
      α : Type u_2
      f : Equiv.Perm α
      inst✝ : DecidableRel f.SameCycle
      x y : α
      h : f.SameCycle x y
      ⊢ f.SameCycle x y
    -/
    exact h
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_2
      f : Equiv.Perm α
      inst✝ : DecidableRel f.SameCycle
      x y : α
      h : Not (f.SameCycle x y)
      ⊢ Eq ((Equiv.Perm.ofSubtype (f.subtypePerm ⋯)) y) y
    -/
  · apply ofSubtype_apply_of_not_mem
    /-
      case neg.ha
      α : Type u_2
      f : Equiv.Perm α
      inst✝ : DecidableRel f.SameCycle
      x y : α
      h : Not (f.SameCycle x y)
      ⊢ Not (f.SameCycle x y)
    -/
    exact h
    /-
      🎉 no goals
    -/


theorem cycleOf_inv (f : Perm α) [DecidableRel f.SameCycle] (x : α) :
    (cycleOf f x)⁻¹ = cycleOf f⁻¹ x :=
  Equiv.ext fun y => by
    /-
      α : Type u_2
      f : Equiv.Perm α
      inst✝ : DecidableRel f.SameCycle
      x y : α
      ⊢ Eq ((Inv.inv (f.cycleOf x)) y) (((Inv.inv f).cycleOf x) y)
    -/
    rw [inv_eq_iff_eq, cycleOf_apply, cycleOf_apply]
    /-
      α : Type u_2
      f : Equiv.Perm α
      inst✝ : DecidableRel f.SameCycle
      x y : α
      ⊢ Eq y (ite (f.SameCycle x (ite ((Inv.inv f).SameCycle x y) ((Inv.inv f) y) y) …
    -/
                  /-
                    🎉 no goals
                  -/
                  /-
                    🎉 no goals
                  -/
                  /-
                    🎉 no goals
                  -/
    split_ifs <;> simp_all [sameCycle_inv, sameCycle_inv_apply_right]
                  /-
                    🎉 no goals
                  -/


@[simp]
theorem cycleOf_pow_apply_self (f : Perm α) [DecidableRel f.SameCycle] (x : α) :
    ∀ n : ℕ, (cycleOf f x ^ n) x = (f ^ n) x := by
  /-
    α : Type u_2
    f : Equiv.Perm α
    inst✝ : DecidableRel f.SameCycle
    x : α
    ⊢ ∀ (n : Nat), Eq ((HPow.hPow (f.cycleOf x) n) x) ((HPow.hPow f n) x)
  -/
  intro n
  induction n with
  | zero => rfl
  | succ n hn =>
    rw [pow_succ', mul_apply, cycleOf_apply, hn, if_pos, pow_succ', mul_apply]
    exact ⟨n, rfl⟩


@[simp]
theorem cycleOf_zpow_apply_self (f : Perm α) [DecidableRel f.SameCycle] (x : α) :
    ∀ n : ℤ, (cycleOf f x ^ n) x = (f ^ n) x := by
  /-
    α : Type u_2
    f : Equiv.Perm α
    inst✝ : DecidableRel f.SameCycle
    x : α
    ⊢ ∀ (n : Int), Eq ((HPow.hPow (f.cycleOf x) n) x) ((HPow.hPow f n) x)
  -/
  intro z
  /-
    α : Type u_2
    f : Equiv.Perm α
    inst✝ : DecidableRel f.SameCycle
    x : α
    z : Int
    ⊢ Eq ((HPow.hPow (f.cycleOf x) z) x) ((HPow.hPow f z) x)
  -/
  induction' z with z hz
    /-
      case ofNat
      α : Type u_2
      f : Equiv.Perm α
      inst✝ : DecidableRel f.SameCycle
      x : α
      z : Nat
      ⊢ Eq ((HPow.hPow (f.cycleOf x) (Int.ofNat z)) x) ((HPow.hPow f (Int.ofNat z)) x)
    -/
  · exact cycleOf_pow_apply_self f x z
    /-
      🎉 no goals
    -/
    /-
      case negSucc
      α : Type u_2
      f : Equiv.Perm α
      inst✝ : DecidableRel f.SameCycle
      x : α
      hz : Nat
      ⊢ Eq ((HPow.hPow (f.cycleOf x) (Int.negSucc hz)) x) ((HPow.hPow f (Int.negSucc …
    -/
  · rw [zpow_negSucc, ← inv_pow, cycleOf_inv, zpow_negSucc, ← inv_pow, cycleOf_pow_apply_self]
    /-
      🎉 no goals
    -/


theorem SameCycle.cycleOf_apply [DecidableRel f.SameCycle] :
    SameCycle f x y → cycleOf f x y = f y :=
  ofSubtype_apply_of_mem _


theorem cycleOf_apply_of_not_sameCycle [DecidableRel f.SameCycle] :
    ¬SameCycle f x y → cycleOf f x y = y :=
  ofSubtype_apply_of_not_mem _


theorem SameCycle.cycleOf_eq [DecidableRel f.SameCycle] (h : SameCycle f x y) :
    cycleOf f x = cycleOf f y := by
  /-
    α : Type u_2
    f : Equiv.Perm α
    x y : α
    inst✝ : DecidableRel f.SameCycle
    h : f.SameCycle x y
    ⊢ Eq (f.cycleOf x) (f.cycleOf y)
  -/
  ext z
  /-
    case H
    α : Type u_2
    f : Equiv.Perm α
    x y : α
    inst✝ : DecidableRel f.SameCycle
    h : f.SameCycle x y
    z : α
    ⊢ Eq ((f.cycleOf x) z) ((f.cycleOf y) z)
  -/
  rw [Equiv.Perm.cycleOf_apply]
  /-
    case H
    α : Type u_2
    f : Equiv.Perm α
    x y : α
    inst✝ : DecidableRel f.SameCycle
    h : f.SameCycle x y
    z : α
    ⊢ Eq (ite (f.SameCycle x z) (f z) z) ((f.cycleOf y) z)
  -/
  split_ifs with hz
    /-
      case pos
      α : Type u_2
      f : Equiv.Perm α
      x y : α
      inst✝ : DecidableRel f.SameCycle
      h : f.SameCycle x y
      z : α
      hz : f.SameCycle x z
      ⊢ Eq (f z) ((f.cycleOf y) z)
    -/
  · exact (h.symm.trans hz).cycleOf_apply.symm
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_2
      f : Equiv.Perm α
      x y : α
      inst✝ : DecidableRel f.SameCycle
      h : f.SameCycle x y
      z : α
      hz : Not (f.SameCycle x z)
      ⊢ Eq z ((f.cycleOf y) z)
    -/
  · exact (cycleOf_apply_of_not_sameCycle (mt h.trans hz)).symm
    /-
      🎉 no goals
    -/


@[simp]
theorem cycleOf_apply_apply_zpow_self (f : Perm α) [DecidableRel f.SameCycle] (x : α) (k : ℤ) :
    cycleOf f x ((f ^ k) x) = (f ^ (k + 1) : Perm α) x := by
  /-
    α : Type u_2
    f : Equiv.Perm α
    inst✝ : DecidableRel f.SameCycle
    x : α
    k : Int
    ⊢ Eq ((f.cycleOf x) ((HPow.hPow f k) x)) ((HPow.hPow f (HAdd.hAdd k 1)) x)
  -/
  rw [SameCycle.cycleOf_apply]
    /-
      α : Type u_2
      f : Equiv.Perm α
      inst✝ : DecidableRel f.SameCycle
      x : α
      k : Int
      ⊢ Eq (f ((HPow.hPow f k) x)) ((HPow.hPow f (HAdd.hAdd k 1)) x)
    -/
  · rw [add_comm, zpow_add, zpow_one, mul_apply]
    /-
      🎉 no goals
    -/
    /-
      α : Type u_2
      f : Equiv.Perm α
      inst✝ : DecidableRel f.SameCycle
      x : α
      k : Int
      ⊢ f.SameCycle x ((HPow.hPow f k) x)
    -/
  · exact ⟨k, rfl⟩
    /-
      🎉 no goals
    -/


@[simp]
theorem cycleOf_apply_apply_pow_self (f : Perm α) [DecidableRel f.SameCycle] (x : α) (k : ℕ) :
    cycleOf f x ((f ^ k) x) = (f ^ (k + 1) : Perm α) x := by
  /-
    α : Type u_2
    f : Equiv.Perm α
    inst✝ : DecidableRel f.SameCycle
    x : α
    k : Nat
    ⊢ Eq ((f.cycleOf x) ((HPow.hPow f k) x)) ((HPow.hPow f (HAdd.hAdd k 1)) x)
  -/
  convert cycleOf_apply_apply_zpow_self f x k using 1
  /-
    🎉 no goals
  -/


@[simp]
theorem cycleOf_apply_apply_self (f : Perm α) [DecidableRel f.SameCycle] (x : α) :
    cycleOf f x (f x) = f (f x) := by
  /-
    α : Type u_2
    f : Equiv.Perm α
    inst✝ : DecidableRel f.SameCycle
    x : α
    ⊢ Eq ((f.cycleOf x) (f x)) (f (f x))
  -/
  convert cycleOf_apply_apply_pow_self f x 1 using 1
  /-
    🎉 no goals
  -/


@[simp]
theorem cycleOf_apply_self (f : Perm α) [DecidableRel f.SameCycle] (x : α) : cycleOf f x x = f x :=
  SameCycle.rfl.cycleOf_apply


theorem IsCycle.cycleOf_eq [DecidableRel f.SameCycle]
    (hf : IsCycle f) (hx : f x ≠ x) : cycleOf f x = f :=
  Equiv.ext fun y =>
                                   /-
                                     α : Type u_2
                                     f : Equiv.Perm α
                                     x : α
                                     inst✝ : DecidableRel f.SameCycle
                                     hf : f.IsCycle
                                     hx : Ne (f x) x
                                     y : α
                                     h : f.SameCycle x y
                                     ⊢ Eq ((f.cycleOf x) y) (f y)
                                   -/
    if h : SameCycle f x y then by rw [h.cycleOf_apply]
                                   /-
                                     🎉 no goals
                                   -/
    else by
      rw [cycleOf_apply_of_not_sameCycle h,
        Classical.not_not.1 (mt ((isCycle_iff_sameCycle hx).1 hf).2 h)]


@[simp]
theorem cycleOf_eq_one_iff (f : Perm α) [DecidableRel f.SameCycle] : cycleOf f x = 1 ↔ f x = x := by
  /-
    α : Type u_2
    x : α
    f : Equiv.Perm α
    inst✝ : DecidableRel f.SameCycle
    ⊢ Iff (Eq (f.cycleOf x) 1) (Eq (f x) x)
  -/
  simp_rw [Perm.ext_iff, cycleOf_apply, one_apply]
  /-
    α : Type u_2
    x : α
    f : Equiv.Perm α
    inst✝ : DecidableRel f.SameCycle
    ⊢ Iff (∀ (x_1 : α), Eq (ite (f.SameCycle x x_1) (f x_1) x_1) x_1) (Eq (f x) x)
  -/
  refine ⟨fun h => (if_pos (SameCycle.refl f x)).symm.trans (h x), fun h y => ?_⟩
  /-
    α : Type u_2
    x : α
    f : Equiv.Perm α
    inst✝ : DecidableRel f.SameCycle
    h : Eq (f x) x
    y : α
    ⊢ Eq (ite (f.SameCycle x y) (f y) y) y
  -/
  by_cases hy : f y = y
    /-
      case pos
      α : Type u_2
      x : α
      f : Equiv.Perm α
      inst✝ : DecidableRel f.SameCycle
      h : Eq (f x) x
      y : α
      hy : Eq (f y) y
      ⊢ Eq (ite (f.SameCycle x y) (f y) y) y
    -/
  · rw [hy, ite_self]
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_2
      x : α
      f : Equiv.Perm α
      inst✝ : DecidableRel f.SameCycle
      h : Eq (f x) x
      y : α
      hy : Not (Eq (f y) y)
      ⊢ Eq (ite (f.SameCycle x y) (f y) y) y
    -/
  · exact if_neg (mt SameCycle.apply_eq_self_iff (by tauto))
    /-
      🎉 no goals
    -/


@[simp]
theorem cycleOf_self_apply (f : Perm α) [DecidableRel f.SameCycle] (x : α) :
    cycleOf f (f x) = cycleOf f x :=
  (sameCycle_apply_right.2 SameCycle.rfl).symm.cycleOf_eq


@[simp]
theorem cycleOf_self_apply_pow (f : Perm α) [DecidableRel f.SameCycle] (n : ℕ) (x : α) :
    cycleOf f ((f ^ n) x) = cycleOf f x :=
  SameCycle.rfl.pow_left.cycleOf_eq


@[simp]
theorem cycleOf_self_apply_zpow (f : Perm α) [DecidableRel f.SameCycle] (n : ℤ) (x : α) :
    cycleOf f ((f ^ n) x) = cycleOf f x :=
  SameCycle.rfl.zpow_left.cycleOf_eq


protected theorem IsCycle.cycleOf [DecidableRel f.SameCycle] [DecidableEq α]
    (hf : IsCycle f) : cycleOf f x = if f x = x then 1 else f := by
  /-
    α : Type u_2
    f : Equiv.Perm α
    x : α
    inst✝¹ : DecidableRel f.SameCycle
    inst✝ : DecidableEq α
    hf : f.IsCycle
    ⊢ Eq (f.cycleOf x) (ite (Eq (f x) x) 1 f)
  -/
  by_cases hx : f x = x
    /-
      case pos
      α : Type u_2
      f : Equiv.Perm α
      x : α
      inst✝¹ : DecidableRel f.SameCycle
      inst✝ : DecidableEq α
      hf : f.IsCycle
      hx : Eq (f x) x
      ⊢ Eq (f.cycleOf x) (ite (Eq (f x) x) 1 f)
    -/
  · rwa [if_pos hx, cycleOf_eq_one_iff]
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_2
      f : Equiv.Perm α
      x : α
      inst✝¹ : DecidableRel f.SameCycle
      inst✝ : DecidableEq α
      hf : f.IsCycle
      hx : Not (Eq (f x) x)
      ⊢ Eq (f.cycleOf x) (ite (Eq (f x) x) 1 f)
    -/
  · rwa [if_neg hx, hf.cycleOf_eq]
    /-
      🎉 no goals
    -/


theorem cycleOf_one [DecidableRel (1 : Perm α).SameCycle] (x : α) :
    cycleOf 1 x = 1 := (cycleOf_eq_one_iff 1).mpr rfl


theorem isCycle_cycleOf (f : Perm α) [DecidableRel f.SameCycle] (hx : f x ≠ x) :
    IsCycle (cycleOf f x) :=
                                 /-
                                   α : Type u_2
                                   x : α
                                   f : Equiv.Perm α
                                   inst✝ : DecidableRel f.SameCycle
                                   hx : Ne (f x) x
                                   ⊢ Ne ((f.cycleOf x) x) x
                                 -/
  have : cycleOf f x x ≠ x := by rwa [SameCycle.rfl.cycleOf_apply]
                                 /-
                                   🎉 no goals
                                 -/
  (isCycle_iff_sameCycle this).2 @fun y =>
    ⟨fun h => mt h.apply_eq_self_iff.2 this, fun h =>
      if hxy : SameCycle f x y then
        let ⟨i, hi⟩ := hxy
               /-
                 α : Type u_2
                 x : α
                 f : Equiv.Perm α
                 inst✝ : DecidableRel f.SameCycle
                 hx : Ne (f x) x
                 this : Ne ((f.cycleOf x) x) x
                 y : α
                 h : Ne ((f.cycleOf x) y) y
                 hxy : f.SameCycle x y
                 i : Int
                 hi : Eq ((HPow.hPow f i) x) y
                 ⊢ Eq ((HPow.hPow (f.cycleOf x) i) x) y
               -/
        ⟨i, by rw [cycleOf_zpow_apply_self, hi]⟩
               /-
                 🎉 no goals
               -/
      else by
        /-
          α : Type u_2
          x : α
          f : Equiv.Perm α
          inst✝ : DecidableRel f.SameCycle
          hx : Ne (f x) x
          this : Ne ((f.cycleOf x) x) x
          y : α
          h : Ne ((f.cycleOf x) y) y
          hxy : Not (f.SameCycle x y)
          ⊢ (f.cycleOf x).SameCycle x y
        -/
        rw [cycleOf_apply_of_not_sameCycle hxy] at h
        /-
          α : Type u_2
          x : α
          f : Equiv.Perm α
          inst✝ : DecidableRel f.SameCycle
          hx : Ne (f x) x
          this : Ne ((f.cycleOf x) x) x
          y : α
          h : Ne y y
          hxy : Not (f.SameCycle x y)
          ⊢ (f.cycleOf x).SameCycle x y
        -/
        exact (h rfl).elim⟩
        /-
          🎉 no goals
        -/


@[simp]
theorem two_le_card_support_cycleOf_iff [DecidableEq α] [Fintype α] :
    2 ≤ #(cycleOf f x).support ↔ f x ≠ x := by
  /-
    α : Type u_2
    f : Equiv.Perm α
    x : α
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    ⊢ Iff (LE.le 2 (f.cycleOf x).support.card) (Ne (f x) x)
  -/
  refine ⟨fun h => ?_, fun h => by simpa using (isCycle_cycleOf _ h).two_le_card_support⟩
  /-
    α : Type u_2
    f : Equiv.Perm α
    x : α
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    h : LE.le 2 (f.cycleOf x).support.card
    ⊢ Ne (f x) x
  -/
  contrapose! h
  /-
    α : Type u_2
    f : Equiv.Perm α
    x : α
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    h : Eq (f x) x
    ⊢ LT.lt (f.cycleOf x).support.card 2
  -/
  rw [← cycleOf_eq_one_iff] at h
  /-
    α : Type u_2
    f : Equiv.Perm α
    x : α
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    h : Eq (f.cycleOf x) 1
    ⊢ LT.lt (f.cycleOf x).support.card 2
  -/
  simp [h]
  /-
    🎉 no goals
  -/


@[simp] lemma support_cycleOf_nonempty [DecidableEq α] [Fintype α] :
    (cycleOf f x).support.Nonempty ↔ f x ≠ x := by
  /-
    α : Type u_2
    f : Equiv.Perm α
    x : α
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    ⊢ Iff (f.cycleOf x).support.Nonempty (Ne (f x) x)
  -/
  rw [← two_le_card_support_cycleOf_iff, ← card_pos, ← Nat.succ_le_iff]
  /-
    α : Type u_2
    f : Equiv.Perm α
    x : α
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    ⊢ Iff (LE.le (Nat.succ 0) (f.cycleOf x).support.card) (LE.le 2 (f.cycleOf x).s …
  -/
  exact ⟨fun h => Or.resolve_left h.eq_or_lt (card_support_ne_one _).symm, zero_lt_two.trans_le⟩
  /-
    🎉 no goals
  -/


@[deprecated support_cycleOf_nonempty (since := "2024-06-16")]
theorem card_support_cycleOf_pos_iff [DecidableEq α] [Fintype α] :
    0 < #(cycleOf f x).support ↔ f x ≠ x := by
  /-
    α : Type u_2
    f : Equiv.Perm α
    x : α
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    ⊢ Iff (LT.lt 0 (f.cycleOf x).support.card) (Ne (f x) x)
  -/
  rw [card_pos, support_cycleOf_nonempty]
  /-
    🎉 no goals
  -/


theorem pow_mod_orderOf_cycleOf_apply (f : Perm α) [DecidableRel f.SameCycle] (n : ℕ) (x : α) :
    (f ^ (n % orderOf (cycleOf f x))) x = (f ^ n) x := by
  /-
    α : Type u_2
    f : Equiv.Perm α
    inst✝ : DecidableRel f.SameCycle
    n : Nat
    x : α
    ⊢ Eq ((HPow.hPow f (HMod.hMod n (orderOf (f.cycleOf x)))) x) ((HPow.hPow f n) x)
  -/
  rw [← cycleOf_pow_apply_self f, ← cycleOf_pow_apply_self f, pow_mod_orderOf]
  /-
    🎉 no goals
  -/


theorem cycleOf_mul_of_apply_right_eq_self [DecidableRel f.SameCycle]
    [DecidableRel (f * g).SameCycle]
    (h : Commute f g) (x : α) (hx : g x = x) : (f * g).cycleOf x = f.cycleOf x := by
  /-
    α : Type u_2
    f g : Equiv.Perm α
    inst✝¹ : DecidableRel f.SameCycle
    inst✝ : DecidableRel (HMul.hMul f g).SameCycle
    h : Commute f g
    x : α
    hx : Eq (g x) x
    ⊢ Eq ((HMul.hMul f g).cycleOf x) (f.cycleOf x)
  -/
  ext y
  /-
    case H
    α : Type u_2
    f g : Equiv.Perm α
    inst✝¹ : DecidableRel f.SameCycle
    inst✝ : DecidableRel (HMul.hMul f g).SameCycle
    h : Commute f g
    x : α
    hx : Eq (g x) x
    y : α
    ⊢ Eq (((HMul.hMul f g).cycleOf x) y) ((f.cycleOf x) y)
  -/
  by_cases hxy : (f * g).SameCycle x y
    /-
      case pos
      α : Type u_2
      f g : Equiv.Perm α
      inst✝¹ : DecidableRel f.SameCycle
      inst✝ : DecidableRel (HMul.hMul f g).SameCycle
      h : Commute f g
      x : α
      hx : Eq (g x) x
      y : α
      hxy : (HMul.hMul f g).SameCycle x y
      ⊢ Eq (((HMul.hMul f g).cycleOf x) y) ((f.cycleOf x) y)
    -/
  · obtain ⟨z, rfl⟩ := hxy
    /-
      case pos.intro
      α : Type u_2
      f g : Equiv.Perm α
      inst✝¹ : DecidableRel f.SameCycle
      inst✝ : DecidableRel (HMul.hMul f g).SameCycle
      h : Commute f g
      x : α
      hx : Eq (g x) x
      z : Int
      ⊢ Eq (((HMul.hMul f g).cycleOf x) ((HPow.hPow (HMul.hMul f g) z) x)) ((f.cycle …
    -/
    rw [cycleOf_apply_apply_zpow_self]
    /-
      case pos.intro
      α : Type u_2
      f g : Equiv.Perm α
      inst✝¹ : DecidableRel f.SameCycle
      inst✝ : DecidableRel (HMul.hMul f g).SameCycle
      h : Commute f g
      x : α
      hx : Eq (g x) x
      z : Int
      ⊢ Eq ((HPow.hPow (HMul.hMul f g) (HAdd.hAdd z 1)) x) ((f.cycleOf x) ((HPow.hPo …
    -/
    simp [h.mul_zpow, zpow_apply_eq_self_of_apply_eq_self hx]
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_2
      f g : Equiv.Perm α
      inst✝¹ : DecidableRel f.SameCycle
      inst✝ : DecidableRel (HMul.hMul f g).SameCycle
      h : Commute f g
      x : α
      hx : Eq (g x) x
      y : α
      hxy : Not ((HMul.hMul f g).SameCycle x y)
      ⊢ Eq (((HMul.hMul f g).cycleOf x) y) ((f.cycleOf x) y)
    -/
  · rw [cycleOf_apply_of_not_sameCycle hxy, cycleOf_apply_of_not_sameCycle]
    /-
      case neg
      α : Type u_2
      f g : Equiv.Perm α
      inst✝¹ : DecidableRel f.SameCycle
      inst✝ : DecidableRel (HMul.hMul f g).SameCycle
      h : Commute f g
      x : α
      hx : Eq (g x) x
      y : α
      hxy : Not ((HMul.hMul f g).SameCycle x y)
      ⊢ Not (f.SameCycle x y)
    -/
    contrapose! hxy
    /-
      case neg
      α : Type u_2
      f g : Equiv.Perm α
      inst✝¹ : DecidableRel f.SameCycle
      inst✝ : DecidableRel (HMul.hMul f g).SameCycle
      h : Commute f g
      x : α
      hx : Eq (g x) x
      y : α
      hxy : f.SameCycle x y
      ⊢ (HMul.hMul f g).SameCycle x y
    -/
    obtain ⟨z, rfl⟩ := hxy
    /-
      case neg.intro
      α : Type u_2
      f g : Equiv.Perm α
      inst✝¹ : DecidableRel f.SameCycle
      inst✝ : DecidableRel (HMul.hMul f g).SameCycle
      h : Commute f g
      x : α
      hx : Eq (g x) x
      z : Int
      ⊢ (HMul.hMul f g).SameCycle x ((HPow.hPow f z) x)
    -/
    refine ⟨z, ?_⟩
    /-
      case neg.intro
      α : Type u_2
      f g : Equiv.Perm α
      inst✝¹ : DecidableRel f.SameCycle
      inst✝ : DecidableRel (HMul.hMul f g).SameCycle
      h : Commute f g
      x : α
      hx : Eq (g x) x
      z : Int
      ⊢ Eq ((HPow.hPow (HMul.hMul f g) z) x) ((HPow.hPow f z) x)
    -/
    simp [h.mul_zpow, zpow_apply_eq_self_of_apply_eq_self hx]
    /-
      🎉 no goals
    -/


theorem Disjoint.cycleOf_mul_distrib [DecidableRel f.SameCycle] [DecidableRel g.SameCycle]
    [DecidableRel (f * g).SameCycle] [DecidableRel (g * f).SameCycle] (h : f.Disjoint g) (x : α) :
    (f * g).cycleOf x = f.cycleOf x * g.cycleOf x := by
  /-
    α : Type u_2
    f g : Equiv.Perm α
    inst✝³ : DecidableRel f.SameCycle
    inst✝² : DecidableRel g.SameCycle
    inst✝¹ : DecidableRel (HMul.hMul f g).SameCycle
    inst✝ : DecidableRel (HMul.hMul g f).SameCycle
    h : f.Disjoint g
    x : α
    ⊢ Eq ((HMul.hMul f g).cycleOf x) (HMul.hMul (f.cycleOf x) (g.cycleOf x))
  -/
  cases' (disjoint_iff_eq_or_eq.mp h) x with hfx hgx
    /-
      case inl
      α : Type u_2
      f g : Equiv.Perm α
      inst✝³ : DecidableRel f.SameCycle
      inst✝² : DecidableRel g.SameCycle
      inst✝¹ : DecidableRel (HMul.hMul f g).SameCycle
      inst✝ : DecidableRel (HMul.hMul g f).SameCycle
      h : f.Disjoint g
      x : α
      hfx : Eq (f x) x
      ⊢ Eq ((HMul.hMul f g).cycleOf x) (HMul.hMul (f.cycleOf x) (g.cycleOf x))
    -/
  · simp [h.commute.eq, cycleOf_mul_of_apply_right_eq_self h.symm.commute, hfx]
    /-
      🎉 no goals
    -/
    /-
      case inr
      α : Type u_2
      f g : Equiv.Perm α
      inst✝³ : DecidableRel f.SameCycle
      inst✝² : DecidableRel g.SameCycle
      inst✝¹ : DecidableRel (HMul.hMul f g).SameCycle
      inst✝ : DecidableRel (HMul.hMul g f).SameCycle
      h : f.Disjoint g
      x : α
      hgx : Eq (g x) x
      ⊢ Eq ((HMul.hMul f g).cycleOf x) (HMul.hMul (f.cycleOf x) (g.cycleOf x))
    -/
  · simp [cycleOf_mul_of_apply_right_eq_self h.commute, hgx]
    /-
      🎉 no goals
    -/


theorem support_cycleOf_eq_nil_iff [DecidableEq α] [Fintype α] :
                                                    /-
                                                      α : Type u_2
                                                      f : Equiv.Perm α
                                                      x : α
                                                      inst✝¹ : DecidableEq α
                                                      inst✝ : Fintype α
                                                      ⊢ Iff (Eq (f.cycleOf x).support EmptyCollection.emptyCollection) (Not (Members …
                                                    -/
    (f.cycleOf x).support = ∅ ↔ x ∉ f.support := by simp
                                                    /-
                                                      🎉 no goals
                                                    -/


theorem support_cycleOf_le [DecidableEq α] [Fintype α] (f : Perm α) (x : α) :
    support (f.cycleOf x) ≤ support f := by
  /-
    α : Type u_2
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    f : Equiv.Perm α
    x : α
    ⊢ LE.le (f.cycleOf x).support f.support
  -/
  intro y hy
  /-
    α : Type u_2
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    f : Equiv.Perm α
    x y : α
    hy : Membership.mem (f.cycleOf x).support y
    ⊢ Membership.mem f.support y
  -/
  rw [mem_support, cycleOf_apply] at hy
  /-
    α : Type u_2
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    f : Equiv.Perm α
    x y : α
    hy : Ne (ite (f.SameCycle x y) (f y) y) y
    ⊢ Membership.mem f.support y
  -/
  split_ifs at hy
    /-
      case pos
      α : Type u_2
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      f : Equiv.Perm α
      x y : α
      h✝ : f.SameCycle x y
      hy : Ne (f y) y
      ⊢ Membership.mem f.support y
    -/
  · exact mem_support.mpr hy
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_2
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      f : Equiv.Perm α
      x y : α
      h✝ : Not (f.SameCycle x y)
      hy : Ne y y
      ⊢ Membership.mem f.support y
    -/
  · exact absurd rfl hy
    /-
      🎉 no goals
    -/


theorem mem_support_cycleOf_iff [DecidableEq α] [Fintype α] :
    y ∈ support (f.cycleOf x) ↔ SameCycle f x y ∧ x ∈ support f := by
  /-
    α : Type u_2
    f : Equiv.Perm α
    x y : α
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    ⊢ Iff (Membership.mem (f.cycleOf x).support y) (And (f.SameCycle x y) (Members …
  -/
  by_cases hx : f x = x
    /-
      case pos
      α : Type u_2
      f : Equiv.Perm α
      x y : α
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      hx : Eq (f x) x
      ⊢ Iff (Membership.mem (f.cycleOf x).support y) (And (f.SameCycle x y) (Members …
    -/
  · rw [(cycleOf_eq_one_iff _).mpr hx]
    /-
      case pos
      α : Type u_2
      f : Equiv.Perm α
      x y : α
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      hx : Eq (f x) x
      ⊢ Iff (Membership.mem (Equiv.Perm.support 1) y) (And (f.SameCycle x y) (Member …
    -/
    simp [hx]
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_2
      f : Equiv.Perm α
      x y : α
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      hx : Not (Eq (f x) x)
      ⊢ Iff (Membership.mem (f.cycleOf x).support y) (And (f.SameCycle x y) (Members …
    -/
  · rw [mem_support, cycleOf_apply]
    /-
      case neg
      α : Type u_2
      f : Equiv.Perm α
      x y : α
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      hx : Not (Eq (f x) x)
      ⊢ Iff (Ne (ite (f.SameCycle x y) (f y) y) y) (And (f.SameCycle x y) (Membershi …
    -/
    split_ifs with hy
      /-
        case pos
        α : Type u_2
        f : Equiv.Perm α
        x y : α
        inst✝¹ : DecidableEq α
        inst✝ : Fintype α
        hx : Not (Eq (f x) x)
        hy : f.SameCycle x y
        ⊢ Iff (Ne (f y) y) (And (f.SameCycle x y) (Membership.mem f.support x))
      -/
    · simp only [hx, hy, Ne, not_false_iff, and_self_iff, mem_support]
      /-
        case pos
        α : Type u_2
        f : Equiv.Perm α
        x y : α
        inst✝¹ : DecidableEq α
        inst✝ : Fintype α
        hx : Not (Eq (f x) x)
        hy : f.SameCycle x y
        ⊢ Iff (Not (Eq (f y) y)) True
      -/
      rcases hy with ⟨k, rfl⟩
      /-
        case pos.intro
        α : Type u_2
        f : Equiv.Perm α
        x : α
        inst✝¹ : DecidableEq α
        inst✝ : Fintype α
        hx : Not (Eq (f x) x)
        k : Int
        ⊢ Iff (Not (Eq (f ((HPow.hPow f k) x)) ((HPow.hPow f k) x))) True
      -/
      rw [← not_mem_support]
      /-
        case pos.intro
        α : Type u_2
        f : Equiv.Perm α
        x : α
        inst✝¹ : DecidableEq α
        inst✝ : Fintype α
        hx : Not (Eq (f x) x)
        k : Int
        ⊢ Iff (Not (Not (Membership.mem f.support ((HPow.hPow f k) x)))) True
      -/
      simpa using hx
      /-
        🎉 no goals
      -/
      /-
        case neg
        α : Type u_2
        f : Equiv.Perm α
        x y : α
        inst✝¹ : DecidableEq α
        inst✝ : Fintype α
        hx : Not (Eq (f x) x)
        hy : Not (f.SameCycle x y)
        ⊢ Iff (Ne y y) (And (f.SameCycle x y) (Membership.mem f.support x))
      -/
    · simpa [hx] using hy
      /-
        🎉 no goals
      -/


theorem mem_support_cycleOf_iff' (hx : f x ≠ x) [DecidableEq α] [Fintype α] :
    y ∈ support (f.cycleOf x) ↔ SameCycle f x y := by
  /-
    α : Type u_2
    f : Equiv.Perm α
    x y : α
    hx : Ne (f x) x
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    ⊢ Iff (Membership.mem (f.cycleOf x).support y) (f.SameCycle x y)
  -/
  rw [mem_support_cycleOf_iff, and_iff_left (mem_support.2 hx)]
  /-
    🎉 no goals
  -/


theorem SameCycle.mem_support_iff {f} [DecidableEq α] [Fintype α] (h : SameCycle f x y) :
    x ∈ support f ↔ y ∈ support f :=
  ⟨fun hx => support_cycleOf_le f x (mem_support_cycleOf_iff.mpr ⟨h, hx⟩), fun hy =>
    support_cycleOf_le f y (mem_support_cycleOf_iff.mpr ⟨h.symm, hy⟩)⟩


theorem pow_mod_card_support_cycleOf_self_apply [DecidableEq α] [Fintype α]
    (f : Perm α) (n : ℕ) (x : α) : (f ^ (n % #(f.cycleOf x).support)) x = (f ^ n) x := by
  /-
    α : Type u_2
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    f : Equiv.Perm α
    n : Nat
    x : α
    ⊢ Eq ((HPow.hPow f (HMod.hMod n (f.cycleOf x).support.card)) x) ((HPow.hPow f  …
  -/
  by_cases hx : f x = x
    /-
      case pos
      α : Type u_2
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      f : Equiv.Perm α
      n : Nat
      x : α
      hx : Eq (f x) x
      ⊢ Eq ((HPow.hPow f (HMod.hMod n (f.cycleOf x).support.card)) x) ((HPow.hPow f  …
    -/
  · rw [pow_apply_eq_self_of_apply_eq_self hx, pow_apply_eq_self_of_apply_eq_self hx]
    /-
      🎉 no goals
    -/
  · rw [← cycleOf_pow_apply_self, ← cycleOf_pow_apply_self f, ← (isCycle_cycleOf f hx).orderOf,
      pow_mod_orderOf]


/-- `x` is in the support of `f` iff `Equiv.Perm.cycle_of f x` is a cycle. -/
theorem isCycle_cycleOf_iff (f : Perm α) [DecidableRel f.SameCycle] :
    IsCycle (cycleOf f x) ↔ f x ≠ x := by
  /-
    α : Type u_2
    x : α
    f : Equiv.Perm α
    inst✝ : DecidableRel f.SameCycle
    ⊢ Iff (f.cycleOf x).IsCycle (Ne (f x) x)
  -/
  refine ⟨fun hx => ?_, f.isCycle_cycleOf⟩
  /-
    α : Type u_2
    x : α
    f : Equiv.Perm α
    inst✝ : DecidableRel f.SameCycle
    hx : (f.cycleOf x).IsCycle
    ⊢ Ne (f x) x
  -/
  rw [Ne, ← cycleOf_eq_one_iff f]
  /-
    α : Type u_2
    x : α
    f : Equiv.Perm α
    inst✝ : DecidableRel f.SameCycle
    hx : (f.cycleOf x).IsCycle
    ⊢ Not (Eq (f.cycleOf x) 1)
  -/
  exact hx.ne_one
  /-
    🎉 no goals
  -/


theorem isCycleOn_support_cycleOf [DecidableEq α] [Fintype α] (f : Perm α) (x : α) :
    f.IsCycleOn (f.cycleOf x).support :=
  ⟨f.bijOn <| by
    /-
      α : Type u_2
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      f : Equiv.Perm α
      x : α
      ⊢ ∀ (a : α), Iff (Membership.mem (↑(f.cycleOf x).support) (f a)) (Membership.m …
    -/
    refine fun _ ↦ ⟨fun h ↦ mem_support_cycleOf_iff.2 ?_, fun h ↦ mem_support_cycleOf_iff.2 ?_⟩
    · exact ⟨sameCycle_apply_right.1 (mem_support_cycleOf_iff.1 h).1,
      (mem_support_cycleOf_iff.1 h).2⟩
    · exact ⟨sameCycle_apply_right.2 (mem_support_cycleOf_iff.1 h).1,
      (mem_support_cycleOf_iff.1 h).2⟩
    , fun a ha b hb =>
      by
        /-
          α : Type u_2
          inst✝¹ : DecidableEq α
          inst✝ : Fintype α
          f : Equiv.Perm α
          x a : α
          ha : Membership.mem (↑(f.cycleOf x).support) a
          b : α
          hb : Membership.mem (↑(f.cycleOf x).support) b
          ⊢ f.SameCycle a b
        -/
        rw [mem_coe, mem_support_cycleOf_iff] at ha hb
        /-
          α : Type u_2
          inst✝¹ : DecidableEq α
          inst✝ : Fintype α
          f : Equiv.Perm α
          x a : α
          ha : And (f.SameCycle x a) (Membership.mem f.support x)
          b : α
          hb : And (f.SameCycle x b) (Membership.mem f.support x)
          ⊢ f.SameCycle a b
        -/
        exact ha.1.symm.trans hb.1⟩
        /-
          🎉 no goals
        -/


theorem SameCycle.exists_pow_eq_of_mem_support {f} [DecidableEq α] [Fintype α] (h : SameCycle f x y)
    (hx : x ∈ f.support) : ∃ i < #(f.cycleOf x).support, (f ^ i) x = y := by
  /-
    α : Type u_2
    x y : α
    f : Equiv.Perm α
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    h : f.SameCycle x y
    hx : Membership.mem f.support x
    ⊢ Exists fun i => And (LT.lt i (f.cycleOf x).support.card) (Eq ((HPow.hPow f i …
  -/
  rw [mem_support] at hx
  exact Equiv.Perm.IsCycleOn.exists_pow_eq (b := y) (f.isCycleOn_support_cycleOf x)
    (by rw [mem_support_cycleOf_iff' hx]) (by rwa [mem_support_cycleOf_iff' hx])


theorem SameCycle.exists_pow_eq [DecidableEq α] [Fintype α] (f : Perm α) (h : SameCycle f x y) :
    ∃ i : ℕ, 0 < i ∧ i ≤ #(f.cycleOf x).support + 1 ∧ (f ^ i) x = y := by
  /-
    α : Type u_2
    x y : α
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    f : Equiv.Perm α
    h : f.SameCycle x y
    ⊢ Exists fun i => And (LT.lt 0 i) (And (LE.le i (HAdd.hAdd (f.cycleOf x).suppo …
  -/
  by_cases hx : x ∈ f.support
    /-
      case pos
      α : Type u_2
      x y : α
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      f : Equiv.Perm α
      h : f.SameCycle x y
      hx : Membership.mem f.support x
      ⊢ Exists fun i => And (LT.lt 0 i) (And (LE.le i (HAdd.hAdd (f.cycleOf x).suppo …
    -/
  · obtain ⟨k, hk, hk'⟩ := h.exists_pow_eq_of_mem_support hx
    /-
      case pos.intro.intro
      α : Type u_2
      x y : α
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      f : Equiv.Perm α
      h : f.SameCycle x y
      hx : Membership.mem f.support x
      k : Nat
      hk : LT.lt k (f.cycleOf x).support.card
      hk' : Eq ((HPow.hPow f k) x) y
      ⊢ Exists fun i => And (LT.lt 0 i) (And (LE.le i (HAdd.hAdd (f.cycleOf x).suppo …
    -/
    cases' k with k
      /-
        case pos.intro.intro.zero
        α : Type u_2
        x y : α
        inst✝¹ : DecidableEq α
        inst✝ : Fintype α
        f : Equiv.Perm α
        h : f.SameCycle x y
        hx : Membership.mem f.support x
        hk : LT.lt 0 (f.cycleOf x).support.card
        hk' : Eq ((HPow.hPow f 0) x) y
        ⊢ Exists fun i => And (LT.lt 0 i) (And (LE.le i (HAdd.hAdd (f.cycleOf x).suppo …
      -/
    · refine ⟨#(f.cycleOf x).support, ?_, self_le_add_right _ _, ?_⟩
        /-
          case pos.intro.intro.zero.refine_1
          α : Type u_2
          x y : α
          inst✝¹ : DecidableEq α
          inst✝ : Fintype α
          f : Equiv.Perm α
          h : f.SameCycle x y
          hx : Membership.mem f.support x
          hk : LT.lt 0 (f.cycleOf x).support.card
          hk' : Eq ((HPow.hPow f 0) x) y
          ⊢ LT.lt 0 (f.cycleOf x).support.card
        -/
      · refine zero_lt_one.trans (one_lt_card_support_of_ne_one ?_)
        /-
          case pos.intro.intro.zero.refine_1
          α : Type u_2
          x y : α
          inst✝¹ : DecidableEq α
          inst✝ : Fintype α
          f : Equiv.Perm α
          h : f.SameCycle x y
          hx : Membership.mem f.support x
          hk : LT.lt 0 (f.cycleOf x).support.card
          hk' : Eq ((HPow.hPow f 0) x) y
          ⊢ Ne (f.cycleOf x) 1
        -/
        simpa using hx
        /-
          🎉 no goals
        -/
        /-
          case pos.intro.intro.zero.refine_2
          α : Type u_2
          x y : α
          inst✝¹ : DecidableEq α
          inst✝ : Fintype α
          f : Equiv.Perm α
          h : f.SameCycle x y
          hx : Membership.mem f.support x
          hk : LT.lt 0 (f.cycleOf x).support.card
          hk' : Eq ((HPow.hPow f 0) x) y
          ⊢ Eq ((HPow.hPow f (f.cycleOf x).support.card) x) y
        -/
      · simp only [pow_zero, coe_one, id_eq] at hk'
        /-
          case pos.intro.intro.zero.refine_2
          α : Type u_2
          x y : α
          inst✝¹ : DecidableEq α
          inst✝ : Fintype α
          f : Equiv.Perm α
          h : f.SameCycle x y
          hx : Membership.mem f.support x
          hk : LT.lt 0 (f.cycleOf x).support.card
          hk' : Eq x y
          ⊢ Eq ((HPow.hPow f (f.cycleOf x).support.card) x) y
        -/
        subst hk'
        rw [← (isCycle_cycleOf _ <| mem_support.1 hx).orderOf, ← cycleOf_pow_apply_self,
          pow_orderOf_eq_one, one_apply]
      /-
        case pos.intro.intro.succ
        α : Type u_2
        x y : α
        inst✝¹ : DecidableEq α
        inst✝ : Fintype α
        f : Equiv.Perm α
        h : f.SameCycle x y
        hx : Membership.mem f.support x
        k : Nat
        hk : LT.lt (HAdd.hAdd k 1) (f.cycleOf x).support.card
        hk' : Eq ((HPow.hPow f (HAdd.hAdd k 1)) x) y
        ⊢ Exists fun i => And (LT.lt 0 i) (And (LE.le i (HAdd.hAdd (f.cycleOf x).suppo …
      -/
    · exact ⟨k + 1, by simp, Nat.le_succ_of_le hk.le, hk'⟩
      /-
        🎉 no goals
      -/
    /-
      case neg
      α : Type u_2
      x y : α
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      f : Equiv.Perm α
      h : f.SameCycle x y
      hx : Not (Membership.mem f.support x)
      ⊢ Exists fun i => And (LT.lt 0 i) (And (LE.le i (HAdd.hAdd (f.cycleOf x).suppo …
    -/
  · refine ⟨1, zero_lt_one, by simp, ?_⟩
    /-
      case neg
      α : Type u_2
      x y : α
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      f : Equiv.Perm α
      h : f.SameCycle x y
      hx : Not (Membership.mem f.support x)
      ⊢ Eq ((HPow.hPow f 1) x) y
    -/
    obtain ⟨k, rfl⟩ := h
    /-
      case neg.intro
      α : Type u_2
      x : α
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      f : Equiv.Perm α
      hx : Not (Membership.mem f.support x)
      k : Int
      ⊢ Eq ((HPow.hPow f 1) x) ((HPow.hPow f k) x)
    -/
    rw [not_mem_support] at hx
    /-
      case neg.intro
      α : Type u_2
      x : α
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      f : Equiv.Perm α
      hx : Eq (f x) x
      k : Int
      ⊢ Eq ((HPow.hPow f 1) x) ((HPow.hPow f k) x)
    -/
    rw [pow_apply_eq_self_of_apply_eq_self hx, zpow_apply_eq_self_of_apply_eq_self hx]
    /-
      🎉 no goals
    -/


theorem zpow_eq_zpow_on_iff [DecidableEq α] [Fintype α]
    (g : Perm α) {m n : ℤ} {x : α} (hx : g x ≠ x) :
    (g ^ m) x = (g ^ n) x ↔ m % #(g.cycleOf x).support = n % #(g.cycleOf x).support := by
  /-
    α : Type u_2
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    g : Equiv.Perm α
    m n : Int
    x : α
    hx : Ne (g x) x
    ⊢ Iff (Eq ((HPow.hPow g m) x) ((HPow.hPow g n) x)) (Eq (HMod.hMod m ↑(g.cycleO …
  -/
  rw [Int.emod_eq_emod_iff_emod_sub_eq_zero]
  /-
    α : Type u_2
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    g : Equiv.Perm α
    m n : Int
    x : α
    hx : Ne (g x) x
    ⊢ Iff (Eq ((HPow.hPow g m) x) ((HPow.hPow g n) x)) (Eq (HMod.hMod (HSub.hSub m …
  -/
  conv_lhs => rw [← Int.sub_add_cancel m n, Int.add_comm, zpow_add]
  /-
    α : Type u_2
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    g : Equiv.Perm α
    m n : Int
    x : α
    hx : Ne (g x) x
    ⊢ Iff (Eq ((HMul.hMul (HPow.hPow g n) (HPow.hPow g (HSub.hSub m n))) x) ((HPow …
  -/
  simp only [coe_mul, Function.comp_apply, EmbeddingLike.apply_eq_iff_eq]
  /-
    α : Type u_2
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    g : Equiv.Perm α
    m n : Int
    x : α
    hx : Ne (g x) x
    ⊢ Iff (Eq ((HPow.hPow g (HSub.hSub m n)) x) x) (Eq (HMod.hMod (HSub.hSub m n)  …
  -/
  rw [← Int.dvd_iff_emod_eq_zero]
  /-
    α : Type u_2
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    g : Equiv.Perm α
    m n : Int
    x : α
    hx : Ne (g x) x
    ⊢ Iff (Eq ((HPow.hPow g (HSub.hSub m n)) x) x) (Dvd.dvd (↑(g.cycleOf x).suppor …
  -/
  rw [← cycleOf_zpow_apply_self g x, cycle_zpow_mem_support_iff]
    /-
      α : Type u_2
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      g : Equiv.Perm α
      m n : Int
      x : α
      hx : Ne (g x) x
      ⊢ Iff (Eq (HMod.hMod (HSub.hSub m n) ↑(g.cycleOf x).support.card) 0) (Dvd.dvd  …
    -/
  · rw [← Int.dvd_iff_emod_eq_zero]
    /-
      🎉 no goals
    -/
    /-
      case hg
      α : Type u_2
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      g : Equiv.Perm α
      m n : Int
      x : α
      hx : Ne (g x) x
      ⊢ (g.cycleOf x).IsCycle
    -/
  · exact isCycle_cycleOf g hx
    /-
      🎉 no goals
    -/
    /-
      case hx
      α : Type u_2
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      g : Equiv.Perm α
      m n : Int
      x : α
      hx : Ne (g x) x
      ⊢ Ne ((g.cycleOf x) x) x
    -/
  · simp only [mem_support, cycleOf_apply_self]; exact hx
                                                 /-
                                                   🎉 no goals
                                                 -/


open scoped List in
/-- Given a list `l : List α` and a permutation `f : Perm α` whose nonfixed points are all in `l`,
  recursively factors `f` into cycles. -/
def cycleFactorsAux [DecidableEq α] [Fintype α] (l : List α) (f : Perm α)
    (h : ∀ {x}, f x ≠ x → x ∈ l) :
    { l : List (Perm α) // l.prod = f ∧ (∀ g ∈ l, IsCycle g) ∧ l.Pairwise Disjoint } :=
  match l with
  | [] => ⟨[], by
      { simp only [imp_false, List.Pairwise.nil, List.not_mem_nil, forall_const, and_true,
          forall_prop_of_false, Classical.not_not, not_false_iff, List.prod_nil] at *
        ext
        simp [*]}⟩
  | x::l =>
    if hx : f x = x then cycleFactorsAux l f (by
        /-
          ι : Type u_1
          α : Type u_2
          β : Type u_3
          inst✝¹ : DecidableEq α
          inst✝ : Fintype α
          l✝ : List α
          f : Equiv.Perm α
          x : α
          l : List α
          h : ∀ {x_1 : α}, Ne (f x_1) x_1 → Membership.mem (List.cons x l) x_1
          hx : Eq (f x) x
          ⊢ ∀ {x : α}, Ne (f x) x → Membership.mem l x
        -/
        intro y hy; exact List.mem_of_ne_of_mem (fun h => hy (by rwa [h])) (h hy))
                    /-
                      🎉 no goals
                    -/
    else
      let ⟨m, hm⟩ :=
        cycleFactorsAux l ((cycleOf f x)⁻¹ * f) (by
        /-
          ι : Type u_1
          α : Type u_2
          β : Type u_3
          inst✝¹ : DecidableEq α
          inst✝ : Fintype α
          l✝ : List α
          f : Equiv.Perm α
          x : α
          l : List α
          h : ∀ {x_1 : α}, Ne (f x_1) x_1 → Membership.mem (List.cons x l) x_1
          hx : Not (Eq (f x) x)
          ⊢ ∀ {x_1 : α}, Ne ((HMul.hMul (Inv.inv (f.cycleOf x)) f) x_1) x_1 → Membership …
        -/
        intro y hy
        exact List.mem_of_ne_of_mem
            (fun h : y = x => by
              rw [h, mul_apply, Ne, inv_eq_iff_eq, cycleOf_apply_self] at hy
              exact hy rfl)
            (h fun h : f y = y => by
              rw [mul_apply, h, Ne, inv_eq_iff_eq, cycleOf_apply] at hy
              split_ifs at hy <;> tauto))
                            /-
                              ι : Type u_1
                              α : Type u_2
                              β : Type u_3
                              inst✝¹ : DecidableEq α
                              inst✝ : Fintype α
                              l✝ : List α
                              f : Equiv.Perm α
                              x : α
                              l : List α
                              h : ∀ {x_1 : α}, Ne (f x_1) x_1 → Membership.mem (List.cons x l) x_1
                              hx : Not (Eq (f x) x)
                              m : List (Equiv.Perm α)
                              hm : And (Eq m.prod (HMul.hMul (Inv.inv (f.cycleOf x)) f)) (And (∀ (g : Equiv. …
                              ⊢ Eq (List.cons (f.cycleOf x) m).prod f
                            -/
      ⟨cycleOf f x :: m, by simp [List.prod_cons, hm.1],
                            /-
                              🎉 no goals
                            -/
        fun g hg ↦ ((List.mem_cons).1 hg).elim (fun hg => hg ▸ isCycle_cycleOf _ hx) (hm.2.1 g),
        List.pairwise_cons.2
          ⟨fun g hg y =>
            or_iff_not_imp_left.2 fun hfy =>
              have hxy : SameCycle f x y :=
                Classical.not_not.1 (mt cycleOf_apply_of_not_sameCycle hfy)
              have hgm : (g::m.erase g) ~ m :=
                List.cons_perm_iff_perm_erase.2 ⟨hg, List.Perm.refl _⟩
              have : ∀ h ∈ m.erase g, Disjoint g h :=
                (List.pairwise_cons.1 ((hgm.pairwise_iff Disjoint.symm).2 hm.2.2)).1
              by_cases id fun hgy : g y ≠ y =>
                (disjoint_prod_right _ this y).resolve_right <| by
                  have hsc : SameCycle f⁻¹ x (f y) := by
                    rwa [sameCycle_inv, sameCycle_apply_right]
                  /-
                    ι : Type u_1
                    α : Type u_2
                    β : Type u_3
                    inst✝¹ : DecidableEq α
                    inst✝ : Fintype α
                    l✝ : List α
                    f : Equiv.Perm α
                    x : α
                    l : List α
                    h : ∀ {x_1 : α}, Ne (f x_1) x_1 → Membership.mem (List.cons x l) x_1
                    hx : Not (Eq (f x) x)
                    m : List (Equiv.Perm α)
                    hm : And (Eq m.prod (HMul.hMul (Inv.inv (f.cycleOf x)) f)) (And (∀ (g : Equiv. …
                    g : Equiv.Perm α
                    hg : Membership.mem m g
                    y : α
                    hfy : Not (Eq ((f.cycleOf x) y) y)
                    hxy : f.SameCycle x y
                    hgm : (List.cons g (m.erase g)).Perm m
                    this : ∀ (h : Equiv.Perm α), Membership.mem (m.erase g) h → g.Disjoint h
                    hgy : Ne (g y) y
                    hsc : (Inv.inv f).SameCycle x (f y)
                    ⊢ Not (Eq ((m.erase g).prod y) y)
                  -/
                  have hm₁ := hm.1
                  rw [disjoint_prod_perm hm.2.2 hgm.symm, List.prod_cons,
                      ← eq_inv_mul_iff_mul_eq] at hm₁
                  rwa [hm₁, mul_apply, mul_apply, cycleOf_inv, hsc.cycleOf_apply, inv_apply_self,
                    inv_eq_iff_eq, eq_comm],
            hm.2.2⟩⟩


theorem mem_list_cycles_iff {α : Type*} [Finite α] {l : List (Perm α)}
    (h1 : ∀ σ : Perm α, σ ∈ l → σ.IsCycle) (h2 : l.Pairwise Disjoint) {σ : Perm α} :
    σ ∈ l ↔ σ.IsCycle ∧ ∀ a, σ a ≠ a → σ a = l.prod a := by
  suffices σ.IsCycle → (σ ∈ l ↔ ∀ a, σ a ≠ a → σ a = l.prod a) by
    exact ⟨fun hσ => ⟨h1 σ hσ, (this (h1 σ hσ)).mp hσ⟩, fun hσ => (this hσ.1).mpr hσ.2⟩
  /-
    α : Type u_4
    inst✝ : Finite α
    l : List (Equiv.Perm α)
    h1 : ∀ (σ : Equiv.Perm α), Membership.mem l σ → σ.IsCycle
    h2 : List.Pairwise Equiv.Perm.Disjoint l
    σ : Equiv.Perm α
    ⊢ σ.IsCycle → Iff (Membership.mem l σ) (∀ (a : α), Ne (σ a) a → Eq (σ a) (l.pr …
  -/
  intro h3
  classical
    cases nonempty_fintype α
    constructor
    · intro h a ha
      exact eq_on_support_mem_disjoint h h2 _ (mem_support.mpr ha)
    · intro h
      have hσl : σ.support ⊆ l.prod.support := by
        intro x hx
        rw [mem_support] at hx
        rwa [mem_support, ← h _ hx]
      obtain ⟨a, ha, -⟩ := id h3
      rw [← mem_support] at ha
      obtain ⟨τ, hτ, hτa⟩ := exists_mem_support_of_mem_support_prod (hσl ha)
      have hτl : ∀ x ∈ τ.support, τ x = l.prod x := eq_on_support_mem_disjoint hτ h2
      have key : ∀ x ∈ σ.support ∩ τ.support, σ x = τ x := by
        intro x hx
        rw [h x (mem_support.mp (mem_of_mem_inter_left hx)), hτl x (mem_of_mem_inter_right hx)]
      convert hτ
      refine h3.eq_on_support_inter_nonempty_congr (h1 _ hτ) key ?_ ha
      exact key a (mem_inter_of_mem ha hτa)


open scoped List in
theorem list_cycles_perm_list_cycles {α : Type*} [Finite α] {l₁ l₂ : List (Perm α)}
    (h₀ : l₁.prod = l₂.prod) (h₁l₁ : ∀ σ : Perm α, σ ∈ l₁ → σ.IsCycle)
    (h₁l₂ : ∀ σ : Perm α, σ ∈ l₂ → σ.IsCycle) (h₂l₁ : l₁.Pairwise Disjoint)
    (h₂l₂ : l₂.Pairwise Disjoint) : l₁ ~ l₂ := by
  classical
    refine
      (List.perm_ext_iff_of_nodup (nodup_of_pairwise_disjoint_cycles h₁l₁ h₂l₁)
            (nodup_of_pairwise_disjoint_cycles h₁l₂ h₂l₂)).mpr
        fun σ => ?_
    by_cases hσ : σ.IsCycle
    · obtain _ := not_forall.mp (mt ext hσ.ne_one)
      rw [mem_list_cycles_iff h₁l₁ h₂l₁, mem_list_cycles_iff h₁l₂ h₂l₂, h₀]
    · exact iff_of_false (mt (h₁l₁ σ) hσ) (mt (h₁l₂ σ) hσ)


/-- Factors a permutation `f` into a list of disjoint cyclic permutations that multiply to `f`. -/
def cycleFactors [Fintype α] [LinearOrder α] (f : Perm α) :
    { l : List (Perm α) // l.prod = f ∧ (∀ g ∈ l, IsCycle g) ∧ l.Pairwise Disjoint } :=
  cycleFactorsAux (sort (α := α) (· ≤ ·) univ) f (fun {_ _} ↦ (mem_sort _).2 (mem_univ _))


/-- Factors a permutation `f` into a list of disjoint cyclic permutations that multiply to `f`,
  without a linear order. -/
def truncCycleFactors [DecidableEq α] [Fintype α] (f : Perm α) :
    Trunc { l : List (Perm α) // l.prod = f ∧ (∀ g ∈ l, IsCycle g) ∧ l.Pairwise Disjoint } :=
  Quotient.recOnSubsingleton (@univ α _).1 (fun l h => Trunc.mk (cycleFactorsAux l f (h _)))
    (show ∀ x, f x ≠ x → x ∈ (@univ α _).1 from fun _ _ => mem_univ _)


/-- Factors a permutation `f` into a `Finset` of disjoint cyclic permutations that multiply to `f`.
-/
def cycleFactorsFinset : Finset (Perm α) :=
  (truncCycleFactors f).lift
    (fun l : { l : List (Perm α) // l.prod = f ∧ (∀ g ∈ l, IsCycle g) ∧ l.Pairwise Disjoint } =>
      l.val.toFinset)
    fun ⟨_, hl⟩ ⟨_, hl'⟩ =>
    List.toFinset_eq_of_perm _ _
      (list_cycles_perm_list_cycles (hl'.left.symm ▸ hl.left) hl.right.left hl'.right.left
        hl.right.right hl'.right.right)


open scoped List in
theorem cycleFactorsFinset_eq_list_toFinset {σ : Perm α} {l : List (Perm α)} (hn : l.Nodup) :
    σ.cycleFactorsFinset = l.toFinset ↔
      (∀ f : Perm α, f ∈ l → f.IsCycle) ∧ l.Pairwise Disjoint ∧ l.prod = σ := by
  /-
    α : Type u_2
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    σ : Equiv.Perm α
    l : List (Equiv.Perm α)
    hn : l.Nodup
    ⊢ Iff (Eq σ.cycleFactorsFinset l.toFinset) (And (∀ (f : Equiv.Perm α), Members …
  -/
  obtain ⟨⟨l', hp', hc', hd'⟩, hl⟩ := Trunc.exists_rep σ.truncCycleFactors
  have ht : cycleFactorsFinset σ = l'.toFinset := by
    rw [cycleFactorsFinset, ← hl, Trunc.lift_mk]
  /-
    case intro.mk.intro.intro
    α : Type u_2
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    σ : Equiv.Perm α
    l : List (Equiv.Perm α)
    hn : l.Nodup
    l' : List (Equiv.Perm α)
    hp' : Eq l'.prod σ
    hc' : ∀ (g : Equiv.Perm α), Membership.mem l' g → g.IsCycle
    hd' : List.Pairwise Equiv.Perm.Disjoint l'
    hl : Eq (Trunc.mk ⟨l', ⋯⟩) σ.truncCycleFactors
    ht : Eq σ.cycleFactorsFinset l'.toFinset
    ⊢ Iff (Eq σ.cycleFactorsFinset l.toFinset) (And (∀ (f : Equiv.Perm α), Members …
  -/
  rw [ht]
  /-
    case intro.mk.intro.intro
    α : Type u_2
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    σ : Equiv.Perm α
    l : List (Equiv.Perm α)
    hn : l.Nodup
    l' : List (Equiv.Perm α)
    hp' : Eq l'.prod σ
    hc' : ∀ (g : Equiv.Perm α), Membership.mem l' g → g.IsCycle
    hd' : List.Pairwise Equiv.Perm.Disjoint l'
    hl : Eq (Trunc.mk ⟨l', ⋯⟩) σ.truncCycleFactors
    ht : Eq σ.cycleFactorsFinset l'.toFinset
    ⊢ Iff (Eq l'.toFinset l.toFinset) (And (∀ (f : Equiv.Perm α), Membership.mem l …
  -/
  constructor
    /-
      case intro.mk.intro.intro.mp
      α : Type u_2
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      σ : Equiv.Perm α
      l : List (Equiv.Perm α)
      hn : l.Nodup
      l' : List (Equiv.Perm α)
      hp' : Eq l'.prod σ
      hc' : ∀ (g : Equiv.Perm α), Membership.mem l' g → g.IsCycle
      hd' : List.Pairwise Equiv.Perm.Disjoint l'
      hl : Eq (Trunc.mk ⟨l', ⋯⟩) σ.truncCycleFactors
      ht : Eq σ.cycleFactorsFinset l'.toFinset
      ⊢ Eq l'.toFinset l.toFinset → And (∀ (f : Equiv.Perm α), Membership.mem l f →  …
    -/
  · intro h
    /-
      case intro.mk.intro.intro.mp
      α : Type u_2
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      σ : Equiv.Perm α
      l : List (Equiv.Perm α)
      hn : l.Nodup
      l' : List (Equiv.Perm α)
      hp' : Eq l'.prod σ
      hc' : ∀ (g : Equiv.Perm α), Membership.mem l' g → g.IsCycle
      hd' : List.Pairwise Equiv.Perm.Disjoint l'
      hl : Eq (Trunc.mk ⟨l', ⋯⟩) σ.truncCycleFactors
      ht : Eq σ.cycleFactorsFinset l'.toFinset
      h : Eq l'.toFinset l.toFinset
      ⊢ And (∀ (f : Equiv.Perm α), Membership.mem l f → f.IsCycle) (And (List.Pairwi …
    -/
    have hn' : l'.Nodup := nodup_of_pairwise_disjoint_cycles hc' hd'
    /-
      case intro.mk.intro.intro.mp
      α : Type u_2
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      σ : Equiv.Perm α
      l : List (Equiv.Perm α)
      hn : l.Nodup
      l' : List (Equiv.Perm α)
      hp' : Eq l'.prod σ
      hc' : ∀ (g : Equiv.Perm α), Membership.mem l' g → g.IsCycle
      hd' : List.Pairwise Equiv.Perm.Disjoint l'
      hl : Eq (Trunc.mk ⟨l', ⋯⟩) σ.truncCycleFactors
      ht : Eq σ.cycleFactorsFinset l'.toFinset
      h : Eq l'.toFinset l.toFinset
      hn' : l'.Nodup
      ⊢ And (∀ (f : Equiv.Perm α), Membership.mem l f → f.IsCycle) (And (List.Pairwi …
    -/
    have hperm : l ~ l' := List.perm_of_nodup_nodup_toFinset_eq hn hn' h.symm
    /-
      case intro.mk.intro.intro.mp
      α : Type u_2
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      σ : Equiv.Perm α
      l : List (Equiv.Perm α)
      hn : l.Nodup
      l' : List (Equiv.Perm α)
      hp' : Eq l'.prod σ
      hc' : ∀ (g : Equiv.Perm α), Membership.mem l' g → g.IsCycle
      hd' : List.Pairwise Equiv.Perm.Disjoint l'
      hl : Eq (Trunc.mk ⟨l', ⋯⟩) σ.truncCycleFactors
      ht : Eq σ.cycleFactorsFinset l'.toFinset
      h : Eq l'.toFinset l.toFinset
      hn' : l'.Nodup
      hperm : l.Perm l'
      ⊢ And (∀ (f : Equiv.Perm α), Membership.mem l f → f.IsCycle) (And (List.Pairwi …
    -/
    refine ⟨?_, ?_, ?_⟩
      /-
        case intro.mk.intro.intro.mp.refine_1
        α : Type u_2
        inst✝¹ : DecidableEq α
        inst✝ : Fintype α
        σ : Equiv.Perm α
        l : List (Equiv.Perm α)
        hn : l.Nodup
        l' : List (Equiv.Perm α)
        hp' : Eq l'.prod σ
        hc' : ∀ (g : Equiv.Perm α), Membership.mem l' g → g.IsCycle
        hd' : List.Pairwise Equiv.Perm.Disjoint l'
        hl : Eq (Trunc.mk ⟨l', ⋯⟩) σ.truncCycleFactors
        ht : Eq σ.cycleFactorsFinset l'.toFinset
        h : Eq l'.toFinset l.toFinset
        hn' : l'.Nodup
        hperm : l.Perm l'
        ⊢ ∀ (f : Equiv.Perm α), Membership.mem l f → f.IsCycle
      -/
    · exact fun _ h => hc' _ (hperm.subset h)
      /-
        🎉 no goals
      -/
      /-
        case intro.mk.intro.intro.mp.refine_2
        α : Type u_2
        inst✝¹ : DecidableEq α
        inst✝ : Fintype α
        σ : Equiv.Perm α
        l : List (Equiv.Perm α)
        hn : l.Nodup
        l' : List (Equiv.Perm α)
        hp' : Eq l'.prod σ
        hc' : ∀ (g : Equiv.Perm α), Membership.mem l' g → g.IsCycle
        hd' : List.Pairwise Equiv.Perm.Disjoint l'
        hl : Eq (Trunc.mk ⟨l', ⋯⟩) σ.truncCycleFactors
        ht : Eq σ.cycleFactorsFinset l'.toFinset
        h : Eq l'.toFinset l.toFinset
        hn' : l'.Nodup
        hperm : l.Perm l'
        ⊢ List.Pairwise Equiv.Perm.Disjoint l
      -/
    · have := List.Perm.pairwise_iff (@Disjoint.symmetric _) hperm
      /-
        case intro.mk.intro.intro.mp.refine_2
        α : Type u_2
        inst✝¹ : DecidableEq α
        inst✝ : Fintype α
        σ : Equiv.Perm α
        l : List (Equiv.Perm α)
        hn : l.Nodup
        l' : List (Equiv.Perm α)
        hp' : Eq l'.prod σ
        hc' : ∀ (g : Equiv.Perm α), Membership.mem l' g → g.IsCycle
        hd' : List.Pairwise Equiv.Perm.Disjoint l'
        hl : Eq (Trunc.mk ⟨l', ⋯⟩) σ.truncCycleFactors
        ht : Eq σ.cycleFactorsFinset l'.toFinset
        h : Eq l'.toFinset l.toFinset
        hn' : l'.Nodup
        hperm : l.Perm l'
        this : Iff (List.Pairwise Equiv.Perm.Disjoint l) (List.Pairwise Equiv.Perm.Dis …
        ⊢ List.Pairwise Equiv.Perm.Disjoint l
      -/
      rwa [this]
      /-
        🎉 no goals
      -/
      /-
        case intro.mk.intro.intro.mp.refine_3
        α : Type u_2
        inst✝¹ : DecidableEq α
        inst✝ : Fintype α
        σ : Equiv.Perm α
        l : List (Equiv.Perm α)
        hn : l.Nodup
        l' : List (Equiv.Perm α)
        hp' : Eq l'.prod σ
        hc' : ∀ (g : Equiv.Perm α), Membership.mem l' g → g.IsCycle
        hd' : List.Pairwise Equiv.Perm.Disjoint l'
        hl : Eq (Trunc.mk ⟨l', ⋯⟩) σ.truncCycleFactors
        ht : Eq σ.cycleFactorsFinset l'.toFinset
        h : Eq l'.toFinset l.toFinset
        hn' : l'.Nodup
        hperm : l.Perm l'
        ⊢ Eq l.prod σ
      -/
    · rw [← hp', hperm.symm.prod_eq']
      /-
        case intro.mk.intro.intro.mp.refine_3
        α : Type u_2
        inst✝¹ : DecidableEq α
        inst✝ : Fintype α
        σ : Equiv.Perm α
        l : List (Equiv.Perm α)
        hn : l.Nodup
        l' : List (Equiv.Perm α)
        hp' : Eq l'.prod σ
        hc' : ∀ (g : Equiv.Perm α), Membership.mem l' g → g.IsCycle
        hd' : List.Pairwise Equiv.Perm.Disjoint l'
        hl : Eq (Trunc.mk ⟨l', ⋯⟩) σ.truncCycleFactors
        ht : Eq σ.cycleFactorsFinset l'.toFinset
        h : Eq l'.toFinset l.toFinset
        hn' : l'.Nodup
        hperm : l.Perm l'
        ⊢ List.Pairwise Commute l'
      -/
      refine hd'.imp ?_
      /-
        case intro.mk.intro.intro.mp.refine_3
        α : Type u_2
        inst✝¹ : DecidableEq α
        inst✝ : Fintype α
        σ : Equiv.Perm α
        l : List (Equiv.Perm α)
        hn : l.Nodup
        l' : List (Equiv.Perm α)
        hp' : Eq l'.prod σ
        hc' : ∀ (g : Equiv.Perm α), Membership.mem l' g → g.IsCycle
        hd' : List.Pairwise Equiv.Perm.Disjoint l'
        hl : Eq (Trunc.mk ⟨l', ⋯⟩) σ.truncCycleFactors
        ht : Eq σ.cycleFactorsFinset l'.toFinset
        h : Eq l'.toFinset l.toFinset
        hn' : l'.Nodup
        hperm : l.Perm l'
        ⊢ ∀ {a b : Equiv.Perm α}, a.Disjoint b → Commute a b
      -/
      exact Disjoint.commute
      /-
        🎉 no goals
      -/
    /-
      case intro.mk.intro.intro.mpr
      α : Type u_2
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      σ : Equiv.Perm α
      l : List (Equiv.Perm α)
      hn : l.Nodup
      l' : List (Equiv.Perm α)
      hp' : Eq l'.prod σ
      hc' : ∀ (g : Equiv.Perm α), Membership.mem l' g → g.IsCycle
      hd' : List.Pairwise Equiv.Perm.Disjoint l'
      hl : Eq (Trunc.mk ⟨l', ⋯⟩) σ.truncCycleFactors
      ht : Eq σ.cycleFactorsFinset l'.toFinset
      ⊢ And (∀ (f : Equiv.Perm α), Membership.mem l f → f.IsCycle) (And (List.Pairwi …
    -/
  · rintro ⟨hc, hd, hp⟩
    /-
      case intro.mk.intro.intro.mpr.intro.intro
      α : Type u_2
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      σ : Equiv.Perm α
      l : List (Equiv.Perm α)
      hn : l.Nodup
      l' : List (Equiv.Perm α)
      hp' : Eq l'.prod σ
      hc' : ∀ (g : Equiv.Perm α), Membership.mem l' g → g.IsCycle
      hd' : List.Pairwise Equiv.Perm.Disjoint l'
      hl : Eq (Trunc.mk ⟨l', ⋯⟩) σ.truncCycleFactors
      ht : Eq σ.cycleFactorsFinset l'.toFinset
      hc : ∀ (f : Equiv.Perm α), Membership.mem l f → f.IsCycle
      hd : List.Pairwise Equiv.Perm.Disjoint l
      hp : Eq l.prod σ
      ⊢ Eq l'.toFinset l.toFinset
    -/
    refine List.toFinset_eq_of_perm _ _ ?_
    /-
      case intro.mk.intro.intro.mpr.intro.intro
      α : Type u_2
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      σ : Equiv.Perm α
      l : List (Equiv.Perm α)
      hn : l.Nodup
      l' : List (Equiv.Perm α)
      hp' : Eq l'.prod σ
      hc' : ∀ (g : Equiv.Perm α), Membership.mem l' g → g.IsCycle
      hd' : List.Pairwise Equiv.Perm.Disjoint l'
      hl : Eq (Trunc.mk ⟨l', ⋯⟩) σ.truncCycleFactors
      ht : Eq σ.cycleFactorsFinset l'.toFinset
      hc : ∀ (f : Equiv.Perm α), Membership.mem l f → f.IsCycle
      hd : List.Pairwise Equiv.Perm.Disjoint l
      hp : Eq l.prod σ
      ⊢ l'.Perm l
    -/
    refine list_cycles_perm_list_cycles ?_ hc' hc hd' hd
    /-
      case intro.mk.intro.intro.mpr.intro.intro
      α : Type u_2
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      σ : Equiv.Perm α
      l : List (Equiv.Perm α)
      hn : l.Nodup
      l' : List (Equiv.Perm α)
      hp' : Eq l'.prod σ
      hc' : ∀ (g : Equiv.Perm α), Membership.mem l' g → g.IsCycle
      hd' : List.Pairwise Equiv.Perm.Disjoint l'
      hl : Eq (Trunc.mk ⟨l', ⋯⟩) σ.truncCycleFactors
      ht : Eq σ.cycleFactorsFinset l'.toFinset
      hc : ∀ (f : Equiv.Perm α), Membership.mem l f → f.IsCycle
      hd : List.Pairwise Equiv.Perm.Disjoint l
      hp : Eq l.prod σ
      ⊢ Eq l'.prod l.prod
    -/
    rw [hp, hp']
    /-
      🎉 no goals
    -/


theorem cycleFactorsFinset_eq_finset {σ : Perm α} {s : Finset (Perm α)} :
    σ.cycleFactorsFinset = s ↔
      (∀ f : Perm α, f ∈ s → f.IsCycle) ∧
        ∃ h : (s : Set (Perm α)).Pairwise Disjoint,
          s.noncommProd id (h.mono' fun _ _ => Disjoint.commute) = σ := by
  /-
    α : Type u_2
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    σ : Equiv.Perm α
    s : Finset (Equiv.Perm α)
    ⊢ Iff (Eq σ.cycleFactorsFinset s) (And (∀ (f : Equiv.Perm α), Membership.mem s …
  -/
  obtain ⟨l, hl, rfl⟩ := s.exists_list_nodup_eq
  /-
    case intro.intro
    α : Type u_2
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    σ : Equiv.Perm α
    l : List (Equiv.Perm α)
    hl : l.Nodup
    ⊢ Iff (Eq σ.cycleFactorsFinset l.toFinset) (And (∀ (f : Equiv.Perm α), Members …
  -/
  simp [cycleFactorsFinset_eq_list_toFinset, hl]
  /-
    🎉 no goals
  -/


theorem cycleFactorsFinset_pairwise_disjoint :
    (cycleFactorsFinset f : Set (Perm α)).Pairwise Disjoint :=
  (cycleFactorsFinset_eq_finset.mp rfl).2.choose


theorem cycleFactorsFinset_mem_commute : (cycleFactorsFinset f : Set (Perm α)).Pairwise Commute :=
  (cycleFactorsFinset_pairwise_disjoint _).mono' fun _ _ => Disjoint.commute


/-- The product of cycle factors is equal to the original `f : perm α`. -/
theorem cycleFactorsFinset_noncommProd
    (comm : (cycleFactorsFinset f : Set (Perm α)).Pairwise Commute :=
      cycleFactorsFinset_mem_commute f) :
    f.cycleFactorsFinset.noncommProd id comm = f :=
  (cycleFactorsFinset_eq_finset.mp rfl).2.choose_spec


theorem mem_cycleFactorsFinset_iff {f p : Perm α} :
    p ∈ cycleFactorsFinset f ↔ p.IsCycle ∧ ∀ a ∈ p.support, p a = f a := by
  /-
    α : Type u_2
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    f p : Equiv.Perm α
    ⊢ Iff (Membership.mem f.cycleFactorsFinset p) (And p.IsCycle (∀ (a : α), Membe …
  -/
  obtain ⟨l, hl, hl'⟩ := f.cycleFactorsFinset.exists_list_nodup_eq
  /-
    case intro.intro
    α : Type u_2
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    f p : Equiv.Perm α
    l : List (Equiv.Perm α)
    hl : l.Nodup
    hl' : Eq l.toFinset f.cycleFactorsFinset
    ⊢ Iff (Membership.mem f.cycleFactorsFinset p) (And p.IsCycle (∀ (a : α), Membe …
  -/
  rw [← hl']
  /-
    case intro.intro
    α : Type u_2
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    f p : Equiv.Perm α
    l : List (Equiv.Perm α)
    hl : l.Nodup
    hl' : Eq l.toFinset f.cycleFactorsFinset
    ⊢ Iff (Membership.mem l.toFinset p) (And p.IsCycle (∀ (a : α), Membership.mem  …
  -/
  rw [eq_comm, cycleFactorsFinset_eq_list_toFinset hl] at hl'
  simpa [List.mem_toFinset, Ne, ← hl'.right.right] using
    mem_list_cycles_iff hl'.left hl'.right.left


theorem cycleOf_mem_cycleFactorsFinset_iff {f : Perm α} {x : α} :
    cycleOf f x ∈ cycleFactorsFinset f ↔ x ∈ f.support := by
  /-
    α : Type u_2
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    f : Equiv.Perm α
    x : α
    ⊢ Iff (Membership.mem f.cycleFactorsFinset (f.cycleOf x)) (Membership.mem f.su …
  -/
  rw [mem_cycleFactorsFinset_iff]
  /-
    α : Type u_2
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    f : Equiv.Perm α
    x : α
    ⊢ Iff (And (f.cycleOf x).IsCycle (∀ (a : α), Membership.mem (f.cycleOf x).supp …
  -/
  constructor
    /-
      case mp
      α : Type u_2
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      f : Equiv.Perm α
      x : α
      ⊢ And (f.cycleOf x).IsCycle (∀ (a : α), Membership.mem (f.cycleOf x).support a …
    -/
  · rintro ⟨hc, _⟩
    /-
      case mp.intro
      α : Type u_2
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      f : Equiv.Perm α
      x : α
      hc : (f.cycleOf x).IsCycle
      right✝ : ∀ (a : α), Membership.mem (f.cycleOf x).support a → Eq ((f.cycleOf x) …
      ⊢ Membership.mem f.support x
    -/
    contrapose! hc
    /-
      case mp.intro
      α : Type u_2
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      f : Equiv.Perm α
      x : α
      right✝ : ∀ (a : α), Membership.mem (f.cycleOf x).support a → Eq ((f.cycleOf x) …
      hc : Not (Membership.mem f.support x)
      ⊢ Not (f.cycleOf x).IsCycle
    -/
    rw [not_mem_support, ← cycleOf_eq_one_iff] at hc
    /-
      case mp.intro
      α : Type u_2
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      f : Equiv.Perm α
      x : α
      right✝ : ∀ (a : α), Membership.mem (f.cycleOf x).support a → Eq ((f.cycleOf x) …
      hc : Eq (f.cycleOf x) 1
      ⊢ Not (f.cycleOf x).IsCycle
    -/
    simp [hc]
    /-
      🎉 no goals
    -/
    /-
      case mpr
      α : Type u_2
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      f : Equiv.Perm α
      x : α
      ⊢ Membership.mem f.support x → And (f.cycleOf x).IsCycle (∀ (a : α), Membershi …
    -/
  · intro hx
    /-
      case mpr
      α : Type u_2
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      f : Equiv.Perm α
      x : α
      hx : Membership.mem f.support x
      ⊢ And (f.cycleOf x).IsCycle (∀ (a : α), Membership.mem (f.cycleOf x).support a …
    -/
    refine ⟨isCycle_cycleOf _ (mem_support.mp hx), ?_⟩
    /-
      case mpr
      α : Type u_2
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      f : Equiv.Perm α
      x : α
      hx : Membership.mem f.support x
      ⊢ ∀ (a : α), Membership.mem (f.cycleOf x).support a → Eq ((f.cycleOf x) a) (f a)
    -/
    intro y hy
    /-
      case mpr
      α : Type u_2
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      f : Equiv.Perm α
      x : α
      hx : Membership.mem f.support x
      y : α
      hy : Membership.mem (f.cycleOf x).support y
      ⊢ Eq ((f.cycleOf x) y) (f y)
    -/
    rw [mem_support] at hy
    /-
      case mpr
      α : Type u_2
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      f : Equiv.Perm α
      x : α
      hx : Membership.mem f.support x
      y : α
      hy : Ne ((f.cycleOf x) y) y
      ⊢ Eq ((f.cycleOf x) y) (f y)
    -/
    rw [cycleOf_apply]
    /-
      case mpr
      α : Type u_2
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      f : Equiv.Perm α
      x : α
      hx : Membership.mem f.support x
      y : α
      hy : Ne ((f.cycleOf x) y) y
      ⊢ Eq (ite (f.SameCycle x y) (f y) y) (f y)
    -/
    split_ifs with H
      /-
        case pos
        α : Type u_2
        inst✝¹ : DecidableEq α
        inst✝ : Fintype α
        f : Equiv.Perm α
        x : α
        hx : Membership.mem f.support x
        y : α
        hy : Ne ((f.cycleOf x) y) y
        H : f.SameCycle x y
        ⊢ Eq (f y) (f y)
      -/
    · rfl
      /-
        🎉 no goals
      -/
      /-
        case neg
        α : Type u_2
        inst✝¹ : DecidableEq α
        inst✝ : Fintype α
        f : Equiv.Perm α
        x : α
        hx : Membership.mem f.support x
        y : α
        hy : Ne ((f.cycleOf x) y) y
        H : Not (f.SameCycle x y)
        ⊢ Eq y (f y)
      -/
    · rw [cycleOf_apply_of_not_sameCycle H] at hy
      /-
        case neg
        α : Type u_2
        inst✝¹ : DecidableEq α
        inst✝ : Fintype α
        f : Equiv.Perm α
        x : α
        hx : Membership.mem f.support x
        y : α
        hy : Ne y y
        H : Not (f.SameCycle x y)
        ⊢ Eq y (f y)
      -/
      contradiction
      /-
        🎉 no goals
      -/


lemma cycleOf_ne_one_iff_mem_cycleFactorsFinset {g : Equiv.Perm α} {x : α} :
    g.cycleOf x ≠ 1 ↔ g.cycleOf x ∈ g.cycleFactorsFinset := by
  /-
    α : Type u_2
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    g : Equiv.Perm α
    x : α
    ⊢ Iff (Ne (g.cycleOf x) 1) (Membership.mem g.cycleFactorsFinset (g.cycleOf x))
  -/
  rw [cycleOf_mem_cycleFactorsFinset_iff, mem_support, ne_eq, cycleOf_eq_one_iff]
  /-
    🎉 no goals
  -/


theorem mem_cycleFactorsFinset_support_le {p f : Perm α} (h : p ∈ cycleFactorsFinset f) :
    p.support ≤ f.support := by
  /-
    α : Type u_2
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    p f : Equiv.Perm α
    h : Membership.mem f.cycleFactorsFinset p
    ⊢ LE.le p.support f.support
  -/
  rw [mem_cycleFactorsFinset_iff] at h
  /-
    α : Type u_2
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    p f : Equiv.Perm α
    h : And p.IsCycle (∀ (a : α), Membership.mem p.support a → Eq (p a) (f a))
    ⊢ LE.le p.support f.support
  -/
  intro x hx
  /-
    α : Type u_2
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    p f : Equiv.Perm α
    h : And p.IsCycle (∀ (a : α), Membership.mem p.support a → Eq (p a) (f a))
    x : α
    hx : Membership.mem p.support x
    ⊢ Membership.mem f.support x
  -/
  rwa [mem_support, ← h.right x hx, ← mem_support]
  /-
    🎉 no goals
  -/


lemma support_zpowers_of_mem_cycleFactorsFinset_le {g : Perm α}
    {c : g.cycleFactorsFinset} (v : Subgroup.zpowers (c : Perm α)) :
    (v : Perm α).support ≤ g.support := by
  /-
    α : Type u_2
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    g : Equiv.Perm α
    c : Subtype fun x => Membership.mem g.cycleFactorsFinset x
    v : Subtype fun x => Membership.mem (Subgroup.zpowers ↑c) x
    ⊢ LE.le (↑v).support g.support
  -/
  obtain ⟨m, hm⟩ := v.prop
  /-
    case intro
    α : Type u_2
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    g : Equiv.Perm α
    c : Subtype fun x => Membership.mem g.cycleFactorsFinset x
    v : Subtype fun x => Membership.mem (Subgroup.zpowers ↑c) x
    m : Int
    hm : Eq ((fun x => HPow.hPow (↑c) x) m) ↑v
    ⊢ LE.le (↑v).support g.support
  -/
  simp only [← hm]
  /-
    case intro
    α : Type u_2
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    g : Equiv.Perm α
    c : Subtype fun x => Membership.mem g.cycleFactorsFinset x
    v : Subtype fun x => Membership.mem (Subgroup.zpowers ↑c) x
    m : Int
    hm : Eq ((fun x => HPow.hPow (↑c) x) m) ↑v
    ⊢ LE.le (HPow.hPow (↑c) m).support g.support
  -/
  exact le_trans (support_zpow_le _ _) (mem_cycleFactorsFinset_support_le c.prop)
  /-
    🎉 no goals
  -/


theorem mem_support_iff_mem_support_of_mem_cycleFactorsFinset {g : Equiv.Perm α} {x : α} :
    x ∈ g.support ↔ ∃ c ∈ g.cycleFactorsFinset, x ∈ c.support := by
  /-
    α : Type u_2
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    g : Equiv.Perm α
    x : α
    ⊢ Iff (Membership.mem g.support x) (Exists fun c => And (Membership.mem g.cycl …
  -/
  constructor
    /-
      case mp
      α : Type u_2
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      g : Equiv.Perm α
      x : α
      ⊢ Membership.mem g.support x → Exists fun c => And (Membership.mem g.cycleFact …
    -/
  · intro h
    /-
      case mp
      α : Type u_2
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      g : Equiv.Perm α
      x : α
      h : Membership.mem g.support x
      ⊢ Exists fun c => And (Membership.mem g.cycleFactorsFinset c) (Membership.mem  …
    -/
    use g.cycleOf x, cycleOf_mem_cycleFactorsFinset_iff.mpr h
    /-
      case right
      α : Type u_2
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      g : Equiv.Perm α
      x : α
      h : Membership.mem g.support x
      ⊢ Membership.mem (g.cycleOf x).support x
    -/
    rw [mem_support_cycleOf_iff]
    /-
      case right
      α : Type u_2
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      g : Equiv.Perm α
      x : α
      h : Membership.mem g.support x
      ⊢ And (g.SameCycle x x) (Membership.mem g.support x)
    -/
    exact ⟨SameCycle.refl g x, h⟩
    /-
      🎉 no goals
    -/
    /-
      case mpr
      α : Type u_2
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      g : Equiv.Perm α
      x : α
      ⊢ (Exists fun c => And (Membership.mem g.cycleFactorsFinset c) (Membership.mem …
    -/
  · rintro ⟨c, hc, hx⟩
    /-
      case mpr.intro.intro
      α : Type u_2
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      g : Equiv.Perm α
      x : α
      c : Equiv.Perm α
      hc : Membership.mem g.cycleFactorsFinset c
      hx : Membership.mem c.support x
      ⊢ Membership.mem g.support x
    -/
    exact mem_cycleFactorsFinset_support_le hc hx
    /-
      🎉 no goals
    -/


theorem cycleFactorsFinset_eq_empty_iff {f : Perm α} : cycleFactorsFinset f = ∅ ↔ f = 1 := by
  /-
    α : Type u_2
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    f : Equiv.Perm α
    ⊢ Iff (Eq f.cycleFactorsFinset EmptyCollection.emptyCollection) (Eq f 1)
  -/
  simpa [cycleFactorsFinset_eq_finset] using eq_comm
  /-
    🎉 no goals
  -/


@[simp]
theorem cycleFactorsFinset_one : cycleFactorsFinset (1 : Perm α) = ∅ := by
  /-
    α : Type u_2
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    ⊢ Eq (Equiv.Perm.cycleFactorsFinset 1) EmptyCollection.emptyCollection
  -/
  simp [cycleFactorsFinset_eq_empty_iff]
  /-
    🎉 no goals
  -/


@[simp]
theorem cycleFactorsFinset_eq_singleton_self_iff {f : Perm α} :
                                                 /-
                                                   α : Type u_2
                                                   inst✝¹ : DecidableEq α
                                                   inst✝ : Fintype α
                                                   f : Equiv.Perm α
                                                   ⊢ Iff (Eq f.cycleFactorsFinset (Singleton.singleton f)) f.IsCycle
                                                 -/
    f.cycleFactorsFinset = {f} ↔ f.IsCycle := by simp [cycleFactorsFinset_eq_finset]
                                                 /-
                                                   🎉 no goals
                                                 -/


theorem IsCycle.cycleFactorsFinset_eq_singleton {f : Perm α} (hf : IsCycle f) :
    f.cycleFactorsFinset = {f} :=
  cycleFactorsFinset_eq_singleton_self_iff.mpr hf


theorem cycleFactorsFinset_eq_singleton_iff {f g : Perm α} :
    f.cycleFactorsFinset = {g} ↔ f.IsCycle ∧ f = g := by
  suffices f = g → (g.IsCycle ↔ f.IsCycle) by
    rw [cycleFactorsFinset_eq_finset]
    simpa [eq_comm]
  /-
    α : Type u_2
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    f g : Equiv.Perm α
    ⊢ Eq f g → Iff g.IsCycle f.IsCycle
  -/
  rintro rfl
  /-
    α : Type u_2
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    f : Equiv.Perm α
    ⊢ Iff f.IsCycle f.IsCycle
  -/
  exact Iff.rfl
  /-
    🎉 no goals
  -/


/-- Two permutations `f g : Perm α` have the same cycle factors iff they are the same. -/
theorem cycleFactorsFinset_injective : Function.Injective (@cycleFactorsFinset α _ _) := by
  /-
    α : Type u_2
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    ⊢ Function.Injective Equiv.Perm.cycleFactorsFinset
  -/
  intro f g h
  /-
    α : Type u_2
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    f g : Equiv.Perm α
    h : Eq f.cycleFactorsFinset g.cycleFactorsFinset
    ⊢ Eq f g
  -/
  rw [← cycleFactorsFinset_noncommProd f]
  /-
    α : Type u_2
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    f g : Equiv.Perm α
    h : Eq f.cycleFactorsFinset g.cycleFactorsFinset
    ⊢ Eq (f.cycleFactorsFinset.noncommProd id ⋯) g
  -/
  simpa [h] using cycleFactorsFinset_noncommProd g
  /-
    🎉 no goals
  -/


theorem Disjoint.disjoint_cycleFactorsFinset {f g : Perm α} (h : Disjoint f g) :
    _root_.Disjoint (cycleFactorsFinset f) (cycleFactorsFinset g) := by
  /-
    α : Type u_2
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    f g : Equiv.Perm α
    h : f.Disjoint g
    ⊢ _root_.Disjoint f.cycleFactorsFinset g.cycleFactorsFinset
  -/
  rw [disjoint_iff_disjoint_support] at h
  /-
    α : Type u_2
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    f g : Equiv.Perm α
    h : _root_.Disjoint f.support g.support
    ⊢ _root_.Disjoint f.cycleFactorsFinset g.cycleFactorsFinset
  -/
  rw [Finset.disjoint_left]
  /-
    α : Type u_2
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    f g : Equiv.Perm α
    h : _root_.Disjoint f.support g.support
    ⊢ ∀ ⦃a : Equiv.Perm α⦄, Membership.mem f.cycleFactorsFinset a → Not (Membershi …
  -/
  intro x hx hy
  /-
    α : Type u_2
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    f g : Equiv.Perm α
    h : _root_.Disjoint f.support g.support
    x : Equiv.Perm α
    hx : Membership.mem f.cycleFactorsFinset x
    hy : Membership.mem g.cycleFactorsFinset x
    ⊢ False
  -/
  simp only [mem_cycleFactorsFinset_iff, mem_support] at hx hy
  /-
    α : Type u_2
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    f g : Equiv.Perm α
    h : _root_.Disjoint f.support g.support
    x : Equiv.Perm α
    hx : And x.IsCycle (∀ (a : α), Ne (x a) a → Eq (x a) (f a))
    hy : And x.IsCycle (∀ (a : α), Ne (x a) a → Eq (x a) (g a))
    ⊢ False
  -/
  obtain ⟨⟨⟨a, ha, -⟩, hf⟩, -, hg⟩ := hx, hy
  /-
    case intro.intro.intro.intro
    α : Type u_2
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    f g : Equiv.Perm α
    h : _root_.Disjoint f.support g.support
    x : Equiv.Perm α
    hf : ∀ (a : α), Ne (x a) a → Eq (x a) (f a)
    a : α
    ha : Ne (x a) a
    hg : ∀ (a : α), Ne (x a) a → Eq (x a) (g a)
    ⊢ False
  -/
  have := h.le_bot (by simp [ha, ← hf a ha, ← hg a ha] : a ∈ f.support ∩ g.support)
  /-
    case intro.intro.intro.intro
    α : Type u_2
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    f g : Equiv.Perm α
    h : _root_.Disjoint f.support g.support
    x : Equiv.Perm α
    hf : ∀ (a : α), Ne (x a) a → Eq (x a) (f a)
    a : α
    ha : Ne (x a) a
    hg : ∀ (a : α), Ne (x a) a → Eq (x a) (g a)
    this : Membership.mem Bot.bot a
    ⊢ False
  -/
  tauto
  /-
    🎉 no goals
  -/


theorem Disjoint.cycleFactorsFinset_mul_eq_union {f g : Perm α} (h : Disjoint f g) :
    cycleFactorsFinset (f * g) = cycleFactorsFinset f ∪ cycleFactorsFinset g := by
  /-
    α : Type u_2
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    f g : Equiv.Perm α
    h : f.Disjoint g
    ⊢ Eq (HMul.hMul f g).cycleFactorsFinset (Union.union f.cycleFactorsFinset g.cy …
  -/
  rw [cycleFactorsFinset_eq_finset]
  /-
    α : Type u_2
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    f g : Equiv.Perm α
    h : f.Disjoint g
    ⊢ And (∀ (f_1 : Equiv.Perm α), Membership.mem (Union.union f.cycleFactorsFinse …
  -/
  refine ⟨?_, ?_, ?_⟩
    /-
      case refine_1
      α : Type u_2
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      f g : Equiv.Perm α
      h : f.Disjoint g
      ⊢ ∀ (f_1 : Equiv.Perm α), Membership.mem (Union.union f.cycleFactorsFinset g.c …
    -/
  · simp [or_imp, mem_cycleFactorsFinset_iff, forall_swap]
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_2
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      f g : Equiv.Perm α
      h : f.Disjoint g
      ⊢ (↑(Union.union f.cycleFactorsFinset g.cycleFactorsFinset)).Pairwise Equiv.Pe …
    -/
  · rw [coe_union, Set.pairwise_union_of_symmetric Disjoint.symmetric]
    exact
      ⟨cycleFactorsFinset_pairwise_disjoint _, cycleFactorsFinset_pairwise_disjoint _,
        fun x hx y hy _ =>
        h.mono (mem_cycleFactorsFinset_support_le hx) (mem_cycleFactorsFinset_support_le hy)⟩
    /-
      case refine_3
      α : Type u_2
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      f g : Equiv.Perm α
      h : f.Disjoint g
      ⊢ Eq ((Union.union f.cycleFactorsFinset g.cycleFactorsFinset).noncommProd id ⋯ …
    -/
  · rw [noncommProd_union_of_disjoint h.disjoint_cycleFactorsFinset]
    /-
      case refine_3
      α : Type u_2
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      f g : Equiv.Perm α
      h : f.Disjoint g
      ⊢ Eq (HMul.hMul (f.cycleFactorsFinset.noncommProd id ⋯) (g.cycleFactorsFinset. …
    -/
    rw [cycleFactorsFinset_noncommProd, cycleFactorsFinset_noncommProd]
    /-
      🎉 no goals
    -/


theorem disjoint_mul_inv_of_mem_cycleFactorsFinset {f g : Perm α} (h : f ∈ cycleFactorsFinset g) :
    Disjoint (g * f⁻¹) f := by
  /-
    α : Type u_2
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    f g : Equiv.Perm α
    h : Membership.mem g.cycleFactorsFinset f
    ⊢ (HMul.hMul g (Inv.inv f)).Disjoint f
  -/
  rw [mem_cycleFactorsFinset_iff] at h
  /-
    α : Type u_2
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    f g : Equiv.Perm α
    h : And f.IsCycle (∀ (a : α), Membership.mem f.support a → Eq (f a) (g a))
    ⊢ (HMul.hMul g (Inv.inv f)).Disjoint f
  -/
  intro x
  /-
    α : Type u_2
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    f g : Equiv.Perm α
    h : And f.IsCycle (∀ (a : α), Membership.mem f.support a → Eq (f a) (g a))
    x : α
    ⊢ Or (Eq ((HMul.hMul g (Inv.inv f)) x) x) (Eq (f x) x)
  -/
  by_cases hx : f x = x
    /-
      case pos
      α : Type u_2
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      f g : Equiv.Perm α
      h : And f.IsCycle (∀ (a : α), Membership.mem f.support a → Eq (f a) (g a))
      x : α
      hx : Eq (f x) x
      ⊢ Or (Eq ((HMul.hMul g (Inv.inv f)) x) x) (Eq (f x) x)
    -/
  · exact Or.inr hx
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_2
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      f g : Equiv.Perm α
      h : And f.IsCycle (∀ (a : α), Membership.mem f.support a → Eq (f a) (g a))
      x : α
      hx : Not (Eq (f x) x)
      ⊢ Or (Eq ((HMul.hMul g (Inv.inv f)) x) x) (Eq (f x) x)
    -/
  · refine Or.inl ?_
    /-
      case neg
      α : Type u_2
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      f g : Equiv.Perm α
      h : And f.IsCycle (∀ (a : α), Membership.mem f.support a → Eq (f a) (g a))
      x : α
      hx : Not (Eq (f x) x)
      ⊢ Eq ((HMul.hMul g (Inv.inv f)) x) x
    -/
    rw [mul_apply, ← h.right, apply_inv_self]
    /-
      case neg.a
      α : Type u_2
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      f g : Equiv.Perm α
      h : And f.IsCycle (∀ (a : α), Membership.mem f.support a → Eq (f a) (g a))
      x : α
      hx : Not (Eq (f x) x)
      ⊢ Membership.mem f.support ((Inv.inv f) x)
    -/
    rwa [← support_inv, apply_mem_support, support_inv, mem_support]
    /-
      🎉 no goals
    -/


/-- If c is a cycle, a ∈ c.support and c is a cycle of f, then `c = f.cycleOf a` -/
theorem cycle_is_cycleOf {f c : Equiv.Perm α} {a : α} (ha : a ∈ c.support)
    (hc : c ∈ f.cycleFactorsFinset) : c = f.cycleOf a := by
  suffices f.cycleOf a = c.cycleOf a by
    rw [this]
    apply symm
    exact
      Equiv.Perm.IsCycle.cycleOf_eq (Equiv.Perm.mem_cycleFactorsFinset_iff.mp hc).left
        (Equiv.Perm.mem_support.mp ha)
  /-
    α : Type u_2
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    f c : Equiv.Perm α
    a : α
    ha : Membership.mem c.support a
    hc : Membership.mem f.cycleFactorsFinset c
    ⊢ Eq (f.cycleOf a) (c.cycleOf a)
  -/
  let hfc := (Equiv.Perm.disjoint_mul_inv_of_mem_cycleFactorsFinset hc).symm
  /-
    α : Type u_2
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    f c : Equiv.Perm α
    a : α
    ha : Membership.mem c.support a
    hc : Membership.mem f.cycleFactorsFinset c
    hfc : c.Disjoint (HMul.hMul f (Inv.inv c)) := Equiv.Perm.Disjoint.symm (Equiv. …
    ⊢ Eq (f.cycleOf a) (c.cycleOf a)
  -/
  let hfc2 := Perm.Disjoint.commute hfc
  /-
    α : Type u_2
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    f c : Equiv.Perm α
    a : α
    ha : Membership.mem c.support a
    hc : Membership.mem f.cycleFactorsFinset c
    hfc : c.Disjoint (HMul.hMul f (Inv.inv c)) := Equiv.Perm.Disjoint.symm (Equiv. …
    hfc2 : Commute c (HMul.hMul f (Inv.inv c)) := Equiv.Perm.Disjoint.commute hfc
    ⊢ Eq (f.cycleOf a) (c.cycleOf a)
  -/
  rw [← Equiv.Perm.cycleOf_mul_of_apply_right_eq_self hfc2]
    /-
      α : Type u_2
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      f c : Equiv.Perm α
      a : α
      ha : Membership.mem c.support a
      hc : Membership.mem f.cycleFactorsFinset c
      hfc : c.Disjoint (HMul.hMul f (Inv.inv c)) := Equiv.Perm.Disjoint.symm (Equiv. …
      hfc2 : Commute c (HMul.hMul f (Inv.inv c)) := Equiv.Perm.Disjoint.commute hfc
      ⊢ Eq (f.cycleOf a) ((HMul.hMul c (HMul.hMul f (Inv.inv c))).cycleOf a)
    -/
  · simp only [hfc2.eq, inv_mul_cancel_right]
    /-
      🎉 no goals
    -/
  -- `a` is in the support of `c`, hence it is not in the support of `g c⁻¹`
  exact
    Equiv.Perm.not_mem_support.mp
      (Finset.disjoint_left.mp (Equiv.Perm.Disjoint.disjoint_support hfc) ha)


theorem isCycleOn_support_of_mem_cycleFactorsFinset {g c : Equiv.Perm α}
    (hc : c ∈ g.cycleFactorsFinset) :
    IsCycleOn g c.support := by
  /-
    α : Type u_2
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    g c : Equiv.Perm α
    hc : Membership.mem g.cycleFactorsFinset c
    ⊢ g.IsCycleOn ↑c.support
  -/
  obtain ⟨x, hx⟩ := IsCycle.nonempty_support (mem_cycleFactorsFinset_iff.mp hc).1
  /-
    case intro
    α : Type u_2
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    g c : Equiv.Perm α
    hc : Membership.mem g.cycleFactorsFinset c
    x : α
    hx : Membership.mem c.support x
    ⊢ g.IsCycleOn ↑c.support
  -/
  rw [cycle_is_cycleOf hx hc]
  /-
    case intro
    α : Type u_2
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    g c : Equiv.Perm α
    hc : Membership.mem g.cycleFactorsFinset c
    x : α
    hx : Membership.mem c.support x
    ⊢ g.IsCycleOn ↑(g.cycleOf x).support
  -/
  exact isCycleOn_support_cycleOf g x
  /-
    🎉 no goals
  -/


theorem eq_cycleOf_of_mem_cycleFactorsFinset_iff
    (g c : Perm α) (hc : c ∈ g.cycleFactorsFinset) (x : α) :
    c = g.cycleOf x ↔ x ∈ c.support := by
  /-
    α : Type u_2
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    g c : Equiv.Perm α
    hc : Membership.mem g.cycleFactorsFinset c
    x : α
    ⊢ Iff (Eq c (g.cycleOf x)) (Membership.mem c.support x)
  -/
  refine ⟨?_, (cycle_is_cycleOf · hc)⟩
  /-
    α : Type u_2
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    g c : Equiv.Perm α
    hc : Membership.mem g.cycleFactorsFinset c
    x : α
    ⊢ Eq c (g.cycleOf x) → Membership.mem c.support x
  -/
  rintro rfl
  /-
    α : Type u_2
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    g : Equiv.Perm α
    x : α
    hc : Membership.mem g.cycleFactorsFinset (g.cycleOf x)
    ⊢ Membership.mem (g.cycleOf x).support x
  -/
  rw [mem_support, cycleOf_apply_self, ne_eq, ← cycleOf_eq_one_iff]
  /-
    α : Type u_2
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    g : Equiv.Perm α
    x : α
    hc : Membership.mem g.cycleFactorsFinset (g.cycleOf x)
    ⊢ Not (Eq (g.cycleOf x) 1)
  -/
  exact (mem_cycleFactorsFinset_iff.mp hc).left.ne_one
  /-
    🎉 no goals
  -/


/-- A permutation `c` is a cycle of `g` iff `k * c * k⁻¹` is a cycle of `k * g * k⁻¹` -/
theorem mem_cycleFactorsFinset_conj (g k c : Perm α) :
    k * c * k⁻¹ ∈ (k * g * k⁻¹).cycleFactorsFinset ↔ c ∈ g.cycleFactorsFinset := by
  suffices imp_lemma : ∀ {g k c : Perm α},
      c ∈ g.cycleFactorsFinset → k * c * k⁻¹ ∈ (k * g * k⁻¹).cycleFactorsFinset by
    refine ⟨fun h ↦ ?_, imp_lemma⟩
    have aux : ∀ h : Perm α, h = k⁻¹ * (k * h * k⁻¹) * k := fun _ ↦ by group
    rw [aux g, aux c]
    exact imp_lemma h
  /-
    α : Type u_2
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    g k c : Equiv.Perm α
    ⊢ ∀ {g k c : Equiv.Perm α}, Membership.mem g.cycleFactorsFinset c → Membership …
  -/
  intro g k c
  /-
    α : Type u_2
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    g✝ k✝ c✝ g k c : Equiv.Perm α
    ⊢ Membership.mem g.cycleFactorsFinset c → Membership.mem (HMul.hMul (HMul.hMul …
  -/
  simp only [mem_cycleFactorsFinset_iff]
  /-
    α : Type u_2
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    g✝ k✝ c✝ g k c : Equiv.Perm α
    ⊢ And c.IsCycle (∀ (a : α), Membership.mem c.support a → Eq (c a) (g a)) → And …
  -/
  apply And.imp IsCycle.conj
  /-
    α : Type u_2
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    g✝ k✝ c✝ g k c : Equiv.Perm α
    ⊢ (∀ (a : α), Membership.mem c.support a → Eq (c a) (g a)) → ∀ (a : α), Member …
  -/
  intro hc a ha
  /-
    α : Type u_2
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    g✝ k✝ c✝ g k c : Equiv.Perm α
    hc : ∀ (a : α), Membership.mem c.support a → Eq (c a) (g a)
    a : α
    ha : Membership.mem (HMul.hMul (HMul.hMul k c) (Inv.inv k)).support a
    ⊢ Eq ((HMul.hMul (HMul.hMul k c) (Inv.inv k)) a) ((HMul.hMul (HMul.hMul k g) ( …
  -/
  simp only [coe_mul, Function.comp_apply, EmbeddingLike.apply_eq_iff_eq]
  /-
    α : Type u_2
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    g✝ k✝ c✝ g k c : Equiv.Perm α
    hc : ∀ (a : α), Membership.mem c.support a → Eq (c a) (g a)
    a : α
    ha : Membership.mem (HMul.hMul (HMul.hMul k c) (Inv.inv k)).support a
    ⊢ Eq (c ((Inv.inv k) a)) (g ((Inv.inv k) a))
  -/
  apply hc
  /-
    case a
    α : Type u_2
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    g✝ k✝ c✝ g k c : Equiv.Perm α
    hc : ∀ (a : α), Membership.mem c.support a → Eq (c a) (g a)
    a : α
    ha : Membership.mem (HMul.hMul (HMul.hMul k c) (Inv.inv k)).support a
    ⊢ Membership.mem c.support ((Inv.inv k) a)
  -/
  rw [mem_support] at ha ⊢
  /-
    case a
    α : Type u_2
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    g✝ k✝ c✝ g k c : Equiv.Perm α
    hc : ∀ (a : α), Membership.mem c.support a → Eq (c a) (g a)
    a : α
    ha : Ne ((HMul.hMul (HMul.hMul k c) (Inv.inv k)) a) a
    ⊢ Ne (c ((Inv.inv k) a)) ((Inv.inv k) a)
  -/
  contrapose! ha
  /-
    case a
    α : Type u_2
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    g✝ k✝ c✝ g k c : Equiv.Perm α
    hc : ∀ (a : α), Membership.mem c.support a → Eq (c a) (g a)
    a : α
    ha : Eq (c ((Inv.inv k) a)) ((Inv.inv k) a)
    ⊢ Eq ((HMul.hMul (HMul.hMul k c) (Inv.inv k)) a) a
  -/
  simp only [mul_smul, ← Perm.smul_def] at ha ⊢
  /-
    case a
    α : Type u_2
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    g✝ k✝ c✝ g k c : Equiv.Perm α
    hc : ∀ (a : α), Membership.mem c.support a → Eq (c a) (g a)
    a : α
    ha : Eq (HSMul.hSMul c (HSMul.hSMul (Inv.inv k) a)) (HSMul.hSMul (Inv.inv k) a)
    ⊢ Eq (HSMul.hSMul k (HSMul.hSMul c (HSMul.hSMul (Inv.inv k) a))) a
  -/
  rw [ha]
  /-
    case a
    α : Type u_2
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    g✝ k✝ c✝ g k c : Equiv.Perm α
    hc : ∀ (a : α), Membership.mem c.support a → Eq (c a) (g a)
    a : α
    ha : Eq (HSMul.hSMul c (HSMul.hSMul (Inv.inv k) a)) (HSMul.hSMul (Inv.inv k) a)
    ⊢ Eq (HSMul.hSMul k (HSMul.hSMul (Inv.inv k) a)) a
  -/
  simp only [Perm.smul_def, apply_inv_self]
  /-
    🎉 no goals
  -/


/-- If a permutation commutes with every cycle of `g`, then it commutes with `g`

NB. The converse is false. Commuting with every cycle of `g` means that we belong
to the kernel of the action of `Equiv.Perm α` on `g.cycleFactorsFinset` -/
theorem commute_of_mem_cycleFactorsFinset_commute (k g : Perm α)
    (hk : ∀ c ∈ g.cycleFactorsFinset, Commute k c) :
    Commute k g := by
  /-
    α : Type u_2
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    k g : Equiv.Perm α
    hk : ∀ (c : Equiv.Perm α), Membership.mem g.cycleFactorsFinset c → Commute k c
    ⊢ Commute k g
  -/
  rw [← cycleFactorsFinset_noncommProd g (cycleFactorsFinset_mem_commute g)]
  /-
    α : Type u_2
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    k g : Equiv.Perm α
    hk : ∀ (c : Equiv.Perm α), Membership.mem g.cycleFactorsFinset c → Commute k c
    ⊢ Commute k (g.cycleFactorsFinset.noncommProd id ⋯)
  -/
  apply Finset.noncommProd_commute
  /-
    case h
    α : Type u_2
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    k g : Equiv.Perm α
    hk : ∀ (c : Equiv.Perm α), Membership.mem g.cycleFactorsFinset c → Commute k c
    ⊢ ∀ (x : Equiv.Perm α), Membership.mem g.cycleFactorsFinset x → Commute k (id x)
  -/
  simpa only [id_eq] using hk
  /-
    🎉 no goals
  -/


/-- The cycles of a permutation commute with it -/
theorem self_mem_cycle_factors_commute {g c : Perm α}
    (hc : c ∈ g.cycleFactorsFinset) : Commute c g := by
  /-
    α : Type u_2
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    g c : Equiv.Perm α
    hc : Membership.mem g.cycleFactorsFinset c
    ⊢ Commute c g
  -/
  apply commute_of_mem_cycleFactorsFinset_commute
  /-
    case hk
    α : Type u_2
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    g c : Equiv.Perm α
    hc : Membership.mem g.cycleFactorsFinset c
    ⊢ ∀ (c_1 : Equiv.Perm α), Membership.mem g.cycleFactorsFinset c_1 → Commute c  …
  -/
  intro c' hc'
  /-
    case hk
    α : Type u_2
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    g c : Equiv.Perm α
    hc : Membership.mem g.cycleFactorsFinset c
    c' : Equiv.Perm α
    hc' : Membership.mem g.cycleFactorsFinset c'
    ⊢ Commute c c'
  -/
  by_cases hcc' : c = c'
    /-
      case pos
      α : Type u_2
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      g c : Equiv.Perm α
      hc : Membership.mem g.cycleFactorsFinset c
      c' : Equiv.Perm α
      hc' : Membership.mem g.cycleFactorsFinset c'
      hcc' : Eq c c'
      ⊢ Commute c c'
    -/
  · rw [hcc']
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_2
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      g c : Equiv.Perm α
      hc : Membership.mem g.cycleFactorsFinset c
      c' : Equiv.Perm α
      hc' : Membership.mem g.cycleFactorsFinset c'
      hcc' : Not (Eq c c')
      ⊢ Commute c c'
    -/
  · apply g.cycleFactorsFinset_mem_commute hc hc'; exact hcc'
                                                   /-
                                                     🎉 no goals
                                                   -/


/-- If `c` and `d` are cycles of `g`, then `d` stabilizes the support of `c` -/
theorem mem_support_cycle_of_cycle {g d c : Perm α}
    (hc : c ∈ g.cycleFactorsFinset) (hd : d ∈ g.cycleFactorsFinset) :
    ∀ x : α, x ∈ c.support ↔ d x ∈ c.support := by
  /-
    α : Type u_2
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    g d c : Equiv.Perm α
    hc : Membership.mem g.cycleFactorsFinset c
    hd : Membership.mem g.cycleFactorsFinset d
    ⊢ ∀ (x : α), Iff (Membership.mem c.support x) (Membership.mem c.support (d x))
  -/
  intro x
  /-
    α : Type u_2
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    g d c : Equiv.Perm α
    hc : Membership.mem g.cycleFactorsFinset c
    hd : Membership.mem g.cycleFactorsFinset d
    x : α
    ⊢ Iff (Membership.mem c.support x) (Membership.mem c.support (d x))
  -/
  simp only [mem_support, not_iff_not]
  /-
    α : Type u_2
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    g d c : Equiv.Perm α
    hc : Membership.mem g.cycleFactorsFinset c
    hd : Membership.mem g.cycleFactorsFinset d
    x : α
    ⊢ Iff (Eq (c x) x) (Eq (c (d x)) (d x))
  -/
  by_cases h : c = d
    /-
      case pos
      α : Type u_2
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      g d c : Equiv.Perm α
      hc : Membership.mem g.cycleFactorsFinset c
      hd : Membership.mem g.cycleFactorsFinset d
      x : α
      h : Eq c d
      ⊢ Iff (Eq (c x) x) (Eq (c (d x)) (d x))
    -/
  · rw [← h, EmbeddingLike.apply_eq_iff_eq]
    /-
      🎉 no goals
    -/
  · rw [← Perm.mul_apply,
      Commute.eq (cycleFactorsFinset_mem_commute g hc hd h),
      mul_apply, EmbeddingLike.apply_eq_iff_eq]


/-- If a permutation is a cycle of `g`, then its support is invariant under `g`-/
theorem mem_cycleFactorsFinset_support {g c : Perm α} (hc : c ∈ g.cycleFactorsFinset) (a : α) :
    a ∈ c.support ↔ g a ∈ c.support :=
  mem_support_iff_of_commute (self_mem_cycle_factors_commute hc).symm a


@[elab_as_elim]
theorem cycle_induction_on [Finite β] (P : Perm β → Prop) (σ : Perm β) (base_one : P 1)
    (base_cycles : ∀ σ : Perm β, σ.IsCycle → P σ)
    (induction_disjoint : ∀ σ τ : Perm β,
      Disjoint σ τ → IsCycle σ → P σ → P τ → P (σ * τ)) : P σ := by
  /-
    β : Type u_3
    inst✝ : Finite β
    P : Equiv.Perm β → Prop
    σ : Equiv.Perm β
    base_one : P 1
    base_cycles : ∀ (σ : Equiv.Perm β), σ.IsCycle → P σ
    induction_disjoint : ∀ (σ τ : Equiv.Perm β), σ.Disjoint τ → σ.IsCycle → P σ →  …
    ⊢ P σ
  -/
  cases nonempty_fintype β
  suffices ∀ l : List (Perm β),
      (∀ τ : Perm β, τ ∈ l → τ.IsCycle) → l.Pairwise Disjoint → P l.prod by
    classical
      let x := σ.truncCycleFactors.out
      exact (congr_arg P x.2.1).mp (this x.1 x.2.2.1 x.2.2.2)
  /-
    case intro
    β : Type u_3
    inst✝ : Finite β
    P : Equiv.Perm β → Prop
    σ : Equiv.Perm β
    base_one : P 1
    base_cycles : ∀ (σ : Equiv.Perm β), σ.IsCycle → P σ
    induction_disjoint : ∀ (σ τ : Equiv.Perm β), σ.Disjoint τ → σ.IsCycle → P σ →  …
    val✝ : Fintype β
    ⊢ ∀ (l : List (Equiv.Perm β)), (∀ (τ : Equiv.Perm β), Membership.mem l τ → τ.I …
  -/
  intro l
  /-
    case intro
    β : Type u_3
    inst✝ : Finite β
    P : Equiv.Perm β → Prop
    σ : Equiv.Perm β
    base_one : P 1
    base_cycles : ∀ (σ : Equiv.Perm β), σ.IsCycle → P σ
    induction_disjoint : ∀ (σ τ : Equiv.Perm β), σ.Disjoint τ → σ.IsCycle → P σ →  …
    val✝ : Fintype β
    l : List (Equiv.Perm β)
    ⊢ (∀ (τ : Equiv.Perm β), Membership.mem l τ → τ.IsCycle) → List.Pairwise Equiv …
  -/
  induction' l with σ l ih
    /-
      case intro.nil
      β : Type u_3
      inst✝ : Finite β
      P : Equiv.Perm β → Prop
      σ : Equiv.Perm β
      base_one : P 1
      base_cycles : ∀ (σ : Equiv.Perm β), σ.IsCycle → P σ
      induction_disjoint : ∀ (σ τ : Equiv.Perm β), σ.Disjoint τ → σ.IsCycle → P σ →  …
      val✝ : Fintype β
      ⊢ (∀ (τ : Equiv.Perm β), Membership.mem List.nil τ → τ.IsCycle) → List.Pairwis …
    -/
  · exact fun _ _ => base_one
    /-
      🎉 no goals
    -/
    /-
      case intro.cons
      β : Type u_3
      inst✝ : Finite β
      P : Equiv.Perm β → Prop
      σ✝ : Equiv.Perm β
      base_one : P 1
      base_cycles : ∀ (σ : Equiv.Perm β), σ.IsCycle → P σ
      induction_disjoint : ∀ (σ τ : Equiv.Perm β), σ.Disjoint τ → σ.IsCycle → P σ →  …
      val✝ : Fintype β
      σ : Equiv.Perm β
      l : List (Equiv.Perm β)
      ih : (∀ (τ : Equiv.Perm β), Membership.mem l τ → τ.IsCycle) → List.Pairwise Eq …
      ⊢ (∀ (τ : Equiv.Perm β), Membership.mem (List.cons σ l) τ → τ.IsCycle) → List. …
    -/
  · intro h1 h2
    /-
      case intro.cons
      β : Type u_3
      inst✝ : Finite β
      P : Equiv.Perm β → Prop
      σ✝ : Equiv.Perm β
      base_one : P 1
      base_cycles : ∀ (σ : Equiv.Perm β), σ.IsCycle → P σ
      induction_disjoint : ∀ (σ τ : Equiv.Perm β), σ.Disjoint τ → σ.IsCycle → P σ →  …
      val✝ : Fintype β
      σ : Equiv.Perm β
      l : List (Equiv.Perm β)
      ih : (∀ (τ : Equiv.Perm β), Membership.mem l τ → τ.IsCycle) → List.Pairwise Eq …
      h1 : ∀ (τ : Equiv.Perm β), Membership.mem (List.cons σ l) τ → τ.IsCycle
      h2 : List.Pairwise Equiv.Perm.Disjoint (List.cons σ l)
      ⊢ P (List.cons σ l).prod
    -/
    rw [List.prod_cons]
    exact
      induction_disjoint σ l.prod (disjoint_prod_right _ (List.pairwise_cons.mp h2).1)
        (h1 _ (List.mem_cons_self _ _)) (base_cycles σ (h1 σ (l.mem_cons_self σ)))
        (ih (fun τ hτ => h1 τ (List.mem_cons_of_mem σ hτ)) h2.of_cons)


theorem cycleFactorsFinset_mul_inv_mem_eq_sdiff [DecidableEq α] [Fintype α] {f g : Perm α}
    (h : f ∈ cycleFactorsFinset g) : cycleFactorsFinset (g * f⁻¹) = cycleFactorsFinset g \ {f} := by
  /-
    α : Type u_2
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    f g : Equiv.Perm α
    h : Membership.mem g.cycleFactorsFinset f
    ⊢ Eq (HMul.hMul g (Inv.inv f)).cycleFactorsFinset (SDiff.sdiff g.cycleFactorsF …
  -/
  revert f
  refine
    cycle_induction_on (P := fun {g : Perm α} ↦
      ∀ {f}, (f ∈ cycleFactorsFinset g)
        → cycleFactorsFinset (g * f⁻¹) = cycleFactorsFinset g \ {f}) _ ?_ ?_ ?_
    /-
      case refine_1
      α : Type u_2
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      g f✝ : Equiv.Perm α
      ⊢ fun {g} => ∀ {f : Equiv.Perm α}, Membership.mem g.cycleFactorsFinset f → Eq  …
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_2
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      g f✝ : Equiv.Perm α
      ⊢ ∀ (σ : Equiv.Perm α), σ.IsCycle → fun {g} => ∀ {f : Equiv.Perm α}, Membershi …
    -/
  · intro σ hσ f hf
    /-
      case refine_2
      α : Type u_2
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      g f✝ σ : Equiv.Perm α
      hσ : σ.IsCycle
      f : Equiv.Perm α
      hf : Membership.mem σ.cycleFactorsFinset f
      ⊢ Eq (HMul.hMul σ (Inv.inv f)).cycleFactorsFinset (SDiff.sdiff σ.cycleFactorsF …
    -/
    simp only [cycleFactorsFinset_eq_singleton_self_iff.mpr hσ, mem_singleton] at hf ⊢
    /-
      case refine_2
      α : Type u_2
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      g f✝ σ : Equiv.Perm α
      hσ : σ.IsCycle
      f : Equiv.Perm α
      hf : Eq f σ
      ⊢ Eq (HMul.hMul σ (Inv.inv f)).cycleFactorsFinset (SDiff.sdiff (Singleton.sing …
    -/
    simp [hf]
    /-
      🎉 no goals
    -/
    /-
      case refine_3
      α : Type u_2
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      g f✝ : Equiv.Perm α
      ⊢ ∀ (σ τ : Equiv.Perm α), σ.Disjoint τ → σ.IsCycle → (fun {g} => ∀ {f : Equiv. …
    -/
  · intro σ τ hd _ hσ hτ f
    /-
      case refine_3
      α : Type u_2
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      g f✝ σ τ : Equiv.Perm α
      hd : σ.Disjoint τ
      a✝ : σ.IsCycle
      hσ : ∀ {f : Equiv.Perm α}, Membership.mem σ.cycleFactorsFinset f → Eq (HMul.hM …
      hτ : ∀ {f : Equiv.Perm α}, Membership.mem τ.cycleFactorsFinset f → Eq (HMul.hM …
      f : Equiv.Perm α
      ⊢ Membership.mem (HMul.hMul σ τ).cycleFactorsFinset f → Eq (HMul.hMul (HMul.hM …
    -/
    simp_rw [hd.cycleFactorsFinset_mul_eq_union, mem_union]
    -- if only `wlog` could work here...
    /-
      case refine_3
      α : Type u_2
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      g f✝ σ τ : Equiv.Perm α
      hd : σ.Disjoint τ
      a✝ : σ.IsCycle
      hσ : ∀ {f : Equiv.Perm α}, Membership.mem σ.cycleFactorsFinset f → Eq (HMul.hM …
      hτ : ∀ {f : Equiv.Perm α}, Membership.mem τ.cycleFactorsFinset f → Eq (HMul.hM …
      f : Equiv.Perm α
      ⊢ Or (Membership.mem σ.cycleFactorsFinset f) (Membership.mem τ.cycleFactorsFin …
    -/
    rintro (hf | hf)
    · rw [hd.commute.eq, union_comm, union_sdiff_distrib, sdiff_singleton_eq_erase,
        erase_eq_of_not_mem, mul_assoc, Disjoint.cycleFactorsFinset_mul_eq_union, hσ hf]
        /-
          case refine_3.inl
          α : Type u_2
          inst✝¹ : DecidableEq α
          inst✝ : Fintype α
          g f✝ σ τ : Equiv.Perm α
          hd : σ.Disjoint τ
          a✝ : σ.IsCycle
          hσ : ∀ {f : Equiv.Perm α}, Membership.mem σ.cycleFactorsFinset f → Eq (HMul.hM …
          hτ : ∀ {f : Equiv.Perm α}, Membership.mem τ.cycleFactorsFinset f → Eq (HMul.hM …
          f : Equiv.Perm α
          hf : Membership.mem σ.cycleFactorsFinset f
          ⊢ τ.Disjoint (HMul.hMul σ (Inv.inv f))
        -/
      · rw [mem_cycleFactorsFinset_iff] at hf
        /-
          case refine_3.inl
          α : Type u_2
          inst✝¹ : DecidableEq α
          inst✝ : Fintype α
          g f✝ σ τ : Equiv.Perm α
          hd : σ.Disjoint τ
          a✝ : σ.IsCycle
          hσ : ∀ {f : Equiv.Perm α}, Membership.mem σ.cycleFactorsFinset f → Eq (HMul.hM …
          hτ : ∀ {f : Equiv.Perm α}, Membership.mem τ.cycleFactorsFinset f → Eq (HMul.hM …
          f : Equiv.Perm α
          hf : And f.IsCycle (∀ (a : α), Membership.mem f.support a → Eq (f a) (σ a))
          ⊢ τ.Disjoint (HMul.hMul σ (Inv.inv f))
        -/
        intro x
        /-
          case refine_3.inl
          α : Type u_2
          inst✝¹ : DecidableEq α
          inst✝ : Fintype α
          g f✝ σ τ : Equiv.Perm α
          hd : σ.Disjoint τ
          a✝ : σ.IsCycle
          hσ : ∀ {f : Equiv.Perm α}, Membership.mem σ.cycleFactorsFinset f → Eq (HMul.hM …
          hτ : ∀ {f : Equiv.Perm α}, Membership.mem τ.cycleFactorsFinset f → Eq (HMul.hM …
          f : Equiv.Perm α
          hf : And f.IsCycle (∀ (a : α), Membership.mem f.support a → Eq (f a) (σ a))
          x : α
          ⊢ Or (Eq (τ x) x) (Eq ((HMul.hMul σ (Inv.inv f)) x) x)
        -/
        cases' hd.symm x with hx hx
          /-
            case refine_3.inl.inl
            α : Type u_2
            inst✝¹ : DecidableEq α
            inst✝ : Fintype α
            g f✝ σ τ : Equiv.Perm α
            hd : σ.Disjoint τ
            a✝ : σ.IsCycle
            hσ : ∀ {f : Equiv.Perm α}, Membership.mem σ.cycleFactorsFinset f → Eq (HMul.hM …
            hτ : ∀ {f : Equiv.Perm α}, Membership.mem τ.cycleFactorsFinset f → Eq (HMul.hM …
            f : Equiv.Perm α
            hf : And f.IsCycle (∀ (a : α), Membership.mem f.support a → Eq (f a) (σ a))
            x : α
            hx : Eq (τ x) x
            ⊢ Or (Eq (τ x) x) (Eq ((HMul.hMul σ (Inv.inv f)) x) x)
          -/
        · exact Or.inl hx
          /-
            🎉 no goals
          -/
          /-
            case refine_3.inl.inr
            α : Type u_2
            inst✝¹ : DecidableEq α
            inst✝ : Fintype α
            g f✝ σ τ : Equiv.Perm α
            hd : σ.Disjoint τ
            a✝ : σ.IsCycle
            hσ : ∀ {f : Equiv.Perm α}, Membership.mem σ.cycleFactorsFinset f → Eq (HMul.hM …
            hτ : ∀ {f : Equiv.Perm α}, Membership.mem τ.cycleFactorsFinset f → Eq (HMul.hM …
            f : Equiv.Perm α
            hf : And f.IsCycle (∀ (a : α), Membership.mem f.support a → Eq (f a) (σ a))
            x : α
            hx : Eq (σ x) x
            ⊢ Or (Eq (τ x) x) (Eq ((HMul.hMul σ (Inv.inv f)) x) x)
          -/
        · refine Or.inr ?_
          /-
            case refine_3.inl.inr
            α : Type u_2
            inst✝¹ : DecidableEq α
            inst✝ : Fintype α
            g f✝ σ τ : Equiv.Perm α
            hd : σ.Disjoint τ
            a✝ : σ.IsCycle
            hσ : ∀ {f : Equiv.Perm α}, Membership.mem σ.cycleFactorsFinset f → Eq (HMul.hM …
            hτ : ∀ {f : Equiv.Perm α}, Membership.mem τ.cycleFactorsFinset f → Eq (HMul.hM …
            f : Equiv.Perm α
            hf : And f.IsCycle (∀ (a : α), Membership.mem f.support a → Eq (f a) (σ a))
            x : α
            hx : Eq (σ x) x
            ⊢ Eq ((HMul.hMul σ (Inv.inv f)) x) x
          -/
          by_cases hfx : f x = x
            /-
              case pos
              α : Type u_2
              inst✝¹ : DecidableEq α
              inst✝ : Fintype α
              g f✝ σ τ : Equiv.Perm α
              hd : σ.Disjoint τ
              a✝ : σ.IsCycle
              hσ : ∀ {f : Equiv.Perm α}, Membership.mem σ.cycleFactorsFinset f → Eq (HMul.hM …
              hτ : ∀ {f : Equiv.Perm α}, Membership.mem τ.cycleFactorsFinset f → Eq (HMul.hM …
              f : Equiv.Perm α
              hf : And f.IsCycle (∀ (a : α), Membership.mem f.support a → Eq (f a) (σ a))
              x : α
              hx : Eq (σ x) x
              hfx : Eq (f x) x
              ⊢ Eq ((HMul.hMul σ (Inv.inv f)) x) x
            -/
          · rw [← hfx]
            /-
              case pos
              α : Type u_2
              inst✝¹ : DecidableEq α
              inst✝ : Fintype α
              g f✝ σ τ : Equiv.Perm α
              hd : σ.Disjoint τ
              a✝ : σ.IsCycle
              hσ : ∀ {f : Equiv.Perm α}, Membership.mem σ.cycleFactorsFinset f → Eq (HMul.hM …
              hτ : ∀ {f : Equiv.Perm α}, Membership.mem τ.cycleFactorsFinset f → Eq (HMul.hM …
              f : Equiv.Perm α
              hf : And f.IsCycle (∀ (a : α), Membership.mem f.support a → Eq (f a) (σ a))
              x : α
              hx : Eq (σ x) x
              hfx : Eq (f x) x
              ⊢ Eq ((HMul.hMul σ (Inv.inv f)) (f x)) (f x)
            -/
            simpa [hx] using hfx.symm
            /-
              🎉 no goals
            -/
            /-
              case neg
              α : Type u_2
              inst✝¹ : DecidableEq α
              inst✝ : Fintype α
              g f✝ σ τ : Equiv.Perm α
              hd : σ.Disjoint τ
              a✝ : σ.IsCycle
              hσ : ∀ {f : Equiv.Perm α}, Membership.mem σ.cycleFactorsFinset f → Eq (HMul.hM …
              hτ : ∀ {f : Equiv.Perm α}, Membership.mem τ.cycleFactorsFinset f → Eq (HMul.hM …
              f : Equiv.Perm α
              hf : And f.IsCycle (∀ (a : α), Membership.mem f.support a → Eq (f a) (σ a))
              x : α
              hx : Eq (σ x) x
              hfx : Not (Eq (f x) x)
              ⊢ Eq ((HMul.hMul σ (Inv.inv f)) x) x
            -/
          · rw [mul_apply]
            /-
              case neg
              α : Type u_2
              inst✝¹ : DecidableEq α
              inst✝ : Fintype α
              g f✝ σ τ : Equiv.Perm α
              hd : σ.Disjoint τ
              a✝ : σ.IsCycle
              hσ : ∀ {f : Equiv.Perm α}, Membership.mem σ.cycleFactorsFinset f → Eq (HMul.hM …
              hτ : ∀ {f : Equiv.Perm α}, Membership.mem τ.cycleFactorsFinset f → Eq (HMul.hM …
              f : Equiv.Perm α
              hf : And f.IsCycle (∀ (a : α), Membership.mem f.support a → Eq (f a) (σ a))
              x : α
              hx : Eq (σ x) x
              hfx : Not (Eq (f x) x)
              ⊢ Eq (σ ((Inv.inv f) x)) x
            -/
            rw [← hf.right _ (mem_support.mpr hfx)] at hx
            /-
              case neg
              α : Type u_2
              inst✝¹ : DecidableEq α
              inst✝ : Fintype α
              g f✝ σ τ : Equiv.Perm α
              hd : σ.Disjoint τ
              a✝ : σ.IsCycle
              hσ : ∀ {f : Equiv.Perm α}, Membership.mem σ.cycleFactorsFinset f → Eq (HMul.hM …
              hτ : ∀ {f : Equiv.Perm α}, Membership.mem τ.cycleFactorsFinset f → Eq (HMul.hM …
              f : Equiv.Perm α
              hf : And f.IsCycle (∀ (a : α), Membership.mem f.support a → Eq (f a) (σ a))
              x : α
              hx : Eq (f x) x
              hfx : Not (Eq (f x) x)
              ⊢ Eq (σ ((Inv.inv f) x)) x
            -/
            contradiction
            /-
              🎉 no goals
            -/
      · exact fun H =>
        not_mem_empty _ (hd.disjoint_cycleFactorsFinset.le_bot (mem_inter_of_mem hf H))
    · rw [union_sdiff_distrib, sdiff_singleton_eq_erase, erase_eq_of_not_mem, mul_assoc,
        Disjoint.cycleFactorsFinset_mul_eq_union, hτ hf]
        /-
          case refine_3.inr
          α : Type u_2
          inst✝¹ : DecidableEq α
          inst✝ : Fintype α
          g f✝ σ τ : Equiv.Perm α
          hd : σ.Disjoint τ
          a✝ : σ.IsCycle
          hσ : ∀ {f : Equiv.Perm α}, Membership.mem σ.cycleFactorsFinset f → Eq (HMul.hM …
          hτ : ∀ {f : Equiv.Perm α}, Membership.mem τ.cycleFactorsFinset f → Eq (HMul.hM …
          f : Equiv.Perm α
          hf : Membership.mem τ.cycleFactorsFinset f
          ⊢ σ.Disjoint (HMul.hMul τ (Inv.inv f))
        -/
      · rw [mem_cycleFactorsFinset_iff] at hf
        /-
          case refine_3.inr
          α : Type u_2
          inst✝¹ : DecidableEq α
          inst✝ : Fintype α
          g f✝ σ τ : Equiv.Perm α
          hd : σ.Disjoint τ
          a✝ : σ.IsCycle
          hσ : ∀ {f : Equiv.Perm α}, Membership.mem σ.cycleFactorsFinset f → Eq (HMul.hM …
          hτ : ∀ {f : Equiv.Perm α}, Membership.mem τ.cycleFactorsFinset f → Eq (HMul.hM …
          f : Equiv.Perm α
          hf : And f.IsCycle (∀ (a : α), Membership.mem f.support a → Eq (f a) (τ a))
          ⊢ σ.Disjoint (HMul.hMul τ (Inv.inv f))
        -/
        intro x
        /-
          case refine_3.inr
          α : Type u_2
          inst✝¹ : DecidableEq α
          inst✝ : Fintype α
          g f✝ σ τ : Equiv.Perm α
          hd : σ.Disjoint τ
          a✝ : σ.IsCycle
          hσ : ∀ {f : Equiv.Perm α}, Membership.mem σ.cycleFactorsFinset f → Eq (HMul.hM …
          hτ : ∀ {f : Equiv.Perm α}, Membership.mem τ.cycleFactorsFinset f → Eq (HMul.hM …
          f : Equiv.Perm α
          hf : And f.IsCycle (∀ (a : α), Membership.mem f.support a → Eq (f a) (τ a))
          x : α
          ⊢ Or (Eq (σ x) x) (Eq ((HMul.hMul τ (Inv.inv f)) x) x)
        -/
        cases' hd x with hx hx
          /-
            case refine_3.inr.inl
            α : Type u_2
            inst✝¹ : DecidableEq α
            inst✝ : Fintype α
            g f✝ σ τ : Equiv.Perm α
            hd : σ.Disjoint τ
            a✝ : σ.IsCycle
            hσ : ∀ {f : Equiv.Perm α}, Membership.mem σ.cycleFactorsFinset f → Eq (HMul.hM …
            hτ : ∀ {f : Equiv.Perm α}, Membership.mem τ.cycleFactorsFinset f → Eq (HMul.hM …
            f : Equiv.Perm α
            hf : And f.IsCycle (∀ (a : α), Membership.mem f.support a → Eq (f a) (τ a))
            x : α
            hx : Eq (σ x) x
            ⊢ Or (Eq (σ x) x) (Eq ((HMul.hMul τ (Inv.inv f)) x) x)
          -/
        · exact Or.inl hx
          /-
            🎉 no goals
          -/
          /-
            case refine_3.inr.inr
            α : Type u_2
            inst✝¹ : DecidableEq α
            inst✝ : Fintype α
            g f✝ σ τ : Equiv.Perm α
            hd : σ.Disjoint τ
            a✝ : σ.IsCycle
            hσ : ∀ {f : Equiv.Perm α}, Membership.mem σ.cycleFactorsFinset f → Eq (HMul.hM …
            hτ : ∀ {f : Equiv.Perm α}, Membership.mem τ.cycleFactorsFinset f → Eq (HMul.hM …
            f : Equiv.Perm α
            hf : And f.IsCycle (∀ (a : α), Membership.mem f.support a → Eq (f a) (τ a))
            x : α
            hx : Eq (τ x) x
            ⊢ Or (Eq (σ x) x) (Eq ((HMul.hMul τ (Inv.inv f)) x) x)
          -/
        · refine Or.inr ?_
          /-
            case refine_3.inr.inr
            α : Type u_2
            inst✝¹ : DecidableEq α
            inst✝ : Fintype α
            g f✝ σ τ : Equiv.Perm α
            hd : σ.Disjoint τ
            a✝ : σ.IsCycle
            hσ : ∀ {f : Equiv.Perm α}, Membership.mem σ.cycleFactorsFinset f → Eq (HMul.hM …
            hτ : ∀ {f : Equiv.Perm α}, Membership.mem τ.cycleFactorsFinset f → Eq (HMul.hM …
            f : Equiv.Perm α
            hf : And f.IsCycle (∀ (a : α), Membership.mem f.support a → Eq (f a) (τ a))
            x : α
            hx : Eq (τ x) x
            ⊢ Eq ((HMul.hMul τ (Inv.inv f)) x) x
          -/
          by_cases hfx : f x = x
            /-
              case pos
              α : Type u_2
              inst✝¹ : DecidableEq α
              inst✝ : Fintype α
              g f✝ σ τ : Equiv.Perm α
              hd : σ.Disjoint τ
              a✝ : σ.IsCycle
              hσ : ∀ {f : Equiv.Perm α}, Membership.mem σ.cycleFactorsFinset f → Eq (HMul.hM …
              hτ : ∀ {f : Equiv.Perm α}, Membership.mem τ.cycleFactorsFinset f → Eq (HMul.hM …
              f : Equiv.Perm α
              hf : And f.IsCycle (∀ (a : α), Membership.mem f.support a → Eq (f a) (τ a))
              x : α
              hx : Eq (τ x) x
              hfx : Eq (f x) x
              ⊢ Eq ((HMul.hMul τ (Inv.inv f)) x) x
            -/
          · rw [← hfx]
            /-
              case pos
              α : Type u_2
              inst✝¹ : DecidableEq α
              inst✝ : Fintype α
              g f✝ σ τ : Equiv.Perm α
              hd : σ.Disjoint τ
              a✝ : σ.IsCycle
              hσ : ∀ {f : Equiv.Perm α}, Membership.mem σ.cycleFactorsFinset f → Eq (HMul.hM …
              hτ : ∀ {f : Equiv.Perm α}, Membership.mem τ.cycleFactorsFinset f → Eq (HMul.hM …
              f : Equiv.Perm α
              hf : And f.IsCycle (∀ (a : α), Membership.mem f.support a → Eq (f a) (τ a))
              x : α
              hx : Eq (τ x) x
              hfx : Eq (f x) x
              ⊢ Eq ((HMul.hMul τ (Inv.inv f)) (f x)) (f x)
            -/
            simpa [hx] using hfx.symm
            /-
              🎉 no goals
            -/
            /-
              case neg
              α : Type u_2
              inst✝¹ : DecidableEq α
              inst✝ : Fintype α
              g f✝ σ τ : Equiv.Perm α
              hd : σ.Disjoint τ
              a✝ : σ.IsCycle
              hσ : ∀ {f : Equiv.Perm α}, Membership.mem σ.cycleFactorsFinset f → Eq (HMul.hM …
              hτ : ∀ {f : Equiv.Perm α}, Membership.mem τ.cycleFactorsFinset f → Eq (HMul.hM …
              f : Equiv.Perm α
              hf : And f.IsCycle (∀ (a : α), Membership.mem f.support a → Eq (f a) (τ a))
              x : α
              hx : Eq (τ x) x
              hfx : Not (Eq (f x) x)
              ⊢ Eq ((HMul.hMul τ (Inv.inv f)) x) x
            -/
          · rw [mul_apply]
            /-
              case neg
              α : Type u_2
              inst✝¹ : DecidableEq α
              inst✝ : Fintype α
              g f✝ σ τ : Equiv.Perm α
              hd : σ.Disjoint τ
              a✝ : σ.IsCycle
              hσ : ∀ {f : Equiv.Perm α}, Membership.mem σ.cycleFactorsFinset f → Eq (HMul.hM …
              hτ : ∀ {f : Equiv.Perm α}, Membership.mem τ.cycleFactorsFinset f → Eq (HMul.hM …
              f : Equiv.Perm α
              hf : And f.IsCycle (∀ (a : α), Membership.mem f.support a → Eq (f a) (τ a))
              x : α
              hx : Eq (τ x) x
              hfx : Not (Eq (f x) x)
              ⊢ Eq (τ ((Inv.inv f) x)) x
            -/
            rw [← hf.right _ (mem_support.mpr hfx)] at hx
            /-
              case neg
              α : Type u_2
              inst✝¹ : DecidableEq α
              inst✝ : Fintype α
              g f✝ σ τ : Equiv.Perm α
              hd : σ.Disjoint τ
              a✝ : σ.IsCycle
              hσ : ∀ {f : Equiv.Perm α}, Membership.mem σ.cycleFactorsFinset f → Eq (HMul.hM …
              hτ : ∀ {f : Equiv.Perm α}, Membership.mem τ.cycleFactorsFinset f → Eq (HMul.hM …
              f : Equiv.Perm α
              hf : And f.IsCycle (∀ (a : α), Membership.mem f.support a → Eq (f a) (τ a))
              x : α
              hx : Eq (f x) x
              hfx : Not (Eq (f x) x)
              ⊢ Eq (τ ((Inv.inv f) x)) x
            -/
            contradiction
            /-
              🎉 no goals
            -/
      · exact fun H =>
        not_mem_empty _ (hd.disjoint_cycleFactorsFinset.le_bot (mem_inter_of_mem H hf))


theorem IsCycle.forall_commute_iff [DecidableEq α] [Fintype α] (g z : Perm α) :
    (∀ c ∈ g.cycleFactorsFinset, Commute z c) ↔
      ∀ c ∈ g.cycleFactorsFinset,
      ∃ (hc : ∀ x : α, x ∈ c.support ↔ z x ∈ c.support),
        ofSubtype (subtypePerm z hc) ∈ Subgroup.zpowers c := by
  /-
    α : Type u_2
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    g z : Equiv.Perm α
    ⊢ Iff (∀ (c : Equiv.Perm α), Membership.mem g.cycleFactorsFinset c → Commute z …
  -/
  apply forall_congr'
  /-
    case h
    α : Type u_2
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    g z : Equiv.Perm α
    ⊢ ∀ (a : Equiv.Perm α), Iff (Membership.mem g.cycleFactorsFinset a → Commute z …
  -/
  intro c
  /-
    case h
    α : Type u_2
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    g z c : Equiv.Perm α
    ⊢ Iff (Membership.mem g.cycleFactorsFinset c → Commute z c) (Membership.mem g. …
  -/
  apply imp_congr_right
  /-
    case h.h
    α : Type u_2
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    g z c : Equiv.Perm α
    ⊢ Membership.mem g.cycleFactorsFinset c → Iff (Commute z c) (Exists fun hc =>  …
  -/
  intro hc
  /-
    case h.h
    α : Type u_2
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    g z c : Equiv.Perm α
    hc : Membership.mem g.cycleFactorsFinset c
    ⊢ Iff (Commute z c) (Exists fun hc => Membership.mem (Subgroup.zpowers c) (Equ …
  -/
  exact IsCycle.commute_iff (mem_cycleFactorsFinset_iff.mp hc).1
  /-
    🎉 no goals
  -/


/-- A permutation restricted to the support of a cycle factor is that cycle factor -/
theorem subtypePerm_on_cycleFactorsFinset [DecidableEq α] [Fintype α]
    {g c : Perm α} (hc : c ∈ g.cycleFactorsFinset) :
    g.subtypePerm (mem_cycleFactorsFinset_support hc) = c.subtypePermOfSupport := by
  /-
    α : Type u_2
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    g c : Equiv.Perm α
    hc : Membership.mem g.cycleFactorsFinset c
    ⊢ Eq (g.subtypePerm ⋯) c.subtypePermOfSupport
  -/
  ext ⟨x, hx⟩
  /-
    case H.mk.a
    α : Type u_2
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    g c : Equiv.Perm α
    hc : Membership.mem g.cycleFactorsFinset c
    x : α
    hx : Membership.mem c.support x
    ⊢ Eq ↑((g.subtypePerm ⋯) ⟨x, hx⟩) ↑(c.subtypePermOfSupport ⟨x, hx⟩)
  -/
  simp only [subtypePerm_apply, Subtype.coe_mk, subtypePermOfSupport]
  /-
    case H.mk.a
    α : Type u_2
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    g c : Equiv.Perm α
    hc : Membership.mem g.cycleFactorsFinset c
    x : α
    hx : Membership.mem c.support x
    ⊢ Eq (g x) (c x)
  -/
  exact ((mem_cycleFactorsFinset_iff.mp hc).2 x hx).symm
  /-
    🎉 no goals
  -/


theorem commute_iff_of_mem_cycleFactorsFinset [DecidableEq α] [Fintype α]{g k c : Equiv.Perm α}
    (hc : c ∈ g.cycleFactorsFinset) :
    Commute k c ↔
      ∃ hc' : ∀ x : α, x ∈ c.support ↔ k x ∈ c.support,
        k.subtypePerm hc' ∈ Subgroup.zpowers
          (g.subtypePerm (mem_cycleFactorsFinset_support hc)) := by
  /-
    α : Type u_2
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    g k c : Equiv.Perm α
    hc : Membership.mem g.cycleFactorsFinset c
    ⊢ Iff (Commute k c) (Exists fun hc' => Membership.mem (Subgroup.zpowers (g.sub …
  -/
  rw [IsCycle.commute_iff' (mem_cycleFactorsFinset_iff.mp hc).1]
  /-
    α : Type u_2
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    g k c : Equiv.Perm α
    hc : Membership.mem g.cycleFactorsFinset c
    ⊢ Iff (Exists fun hc' => Membership.mem (Subgroup.zpowers c.subtypePermOfSuppo …
  -/
  apply exists_congr
  /-
    case h
    α : Type u_2
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    g k c : Equiv.Perm α
    hc : Membership.mem g.cycleFactorsFinset c
    ⊢ ∀ (a : ∀ (x : α), Iff (Membership.mem c.support x) (Membership.mem c.support …
  -/
  intro hc'
  /-
    case h
    α : Type u_2
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    g k c : Equiv.Perm α
    hc : Membership.mem g.cycleFactorsFinset c
    hc' : ∀ (x : α), Iff (Membership.mem c.support x) (Membership.mem c.support (k …
    ⊢ Iff (Membership.mem (Subgroup.zpowers c.subtypePermOfSupport) (k.subtypePerm …
  -/
  simp only [Subgroup.mem_zpowers_iff]
  /-
    case h
    α : Type u_2
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    g k c : Equiv.Perm α
    hc : Membership.mem g.cycleFactorsFinset c
    hc' : ∀ (x : α), Iff (Membership.mem c.support x) (Membership.mem c.support (k …
    ⊢ Iff (Exists fun k_1 => Eq (HPow.hPow c.subtypePermOfSupport k_1) (k.subtypeP …
  -/
  apply exists_congr
  /-
    case h.h
    α : Type u_2
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    g k c : Equiv.Perm α
    hc : Membership.mem g.cycleFactorsFinset c
    hc' : ∀ (x : α), Iff (Membership.mem c.support x) (Membership.mem c.support (k …
    ⊢ ∀ (a : Int), Iff (Eq (HPow.hPow c.subtypePermOfSupport a) (k.subtypePerm hc' …
  -/
  intro n
  /-
    case h.h
    α : Type u_2
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    g k c : Equiv.Perm α
    hc : Membership.mem g.cycleFactorsFinset c
    hc' : ∀ (x : α), Iff (Membership.mem c.support x) (Membership.mem c.support (k …
    n : Int
    ⊢ Iff (Eq (HPow.hPow c.subtypePermOfSupport n) (k.subtypePerm hc')) (Eq (HPow. …
  -/
  rw [Equiv.Perm.subtypePerm_on_cycleFactorsFinset hc]
  /-
    🎉 no goals
  -/


