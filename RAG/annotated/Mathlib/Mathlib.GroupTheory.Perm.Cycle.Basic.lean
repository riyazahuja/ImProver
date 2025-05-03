/-- The equivalence relation indicating that two points are in the same cycle of a permutation. -/
def SameCycle (f : Perm α) (x y : α) : Prop :=
  ∃ i : ℤ, (f ^ i) x = y


@[refl]
theorem SameCycle.refl (f : Perm α) (x : α) : SameCycle f x x :=
  ⟨0, rfl⟩


theorem SameCycle.rfl : SameCycle f x x :=
  SameCycle.refl _ _


                                                                                       /-
                                                                                         α : Type u_2
                                                                                         x y : α
                                                                                         h : Eq x y
                                                                                         f : Equiv.Perm α
                                                                                         ⊢ f.SameCycle x y
                                                                                       -/
protected theorem _root_.Eq.sameCycle (h : x = y) (f : Perm α) : f.SameCycle x y := by rw [h]
                                                                                       /-
                                                                                         🎉 no goals
                                                                                       -/


@[symm]
theorem SameCycle.symm : SameCycle f x y → SameCycle f y x := fun ⟨i, hi⟩ =>
          /-
            α : Type u_2
            f : Equiv.Perm α
            x y : α
            x✝ : f.SameCycle x y
            i : Int
            hi : Eq ((HPow.hPow f i) x) y
            ⊢ Eq ((HPow.hPow f (Neg.neg i)) y) x
          -/
  ⟨-i, by rw [zpow_neg, ← hi, inv_apply_self]⟩
          /-
            🎉 no goals
          -/


theorem sameCycle_comm : SameCycle f x y ↔ SameCycle f y x :=
  ⟨SameCycle.symm, SameCycle.symm⟩


@[trans]
theorem SameCycle.trans : SameCycle f x y → SameCycle f y z → SameCycle f x z :=
                                    /-
                                      α : Type u_2
                                      f : Equiv.Perm α
                                      x y z : α
                                      x✝¹ : f.SameCycle x y
                                      x✝ : f.SameCycle y z
                                      i : Int
                                      hi : Eq ((HPow.hPow f i) x) y
                                      j : Int
                                      hj : Eq ((HPow.hPow f j) y) z
                                      ⊢ Eq ((HPow.hPow f (HAdd.hAdd j i)) x) z
                                    -/
  fun ⟨i, hi⟩ ⟨j, hj⟩ => ⟨j + i, by rw [zpow_add, mul_apply, hi, hj]⟩
                                    /-
                                      🎉 no goals
                                    -/


variable (f) in
theorem SameCycle.equivalence : Equivalence (SameCycle f) :=
  ⟨SameCycle.refl f, SameCycle.symm, SameCycle.trans⟩


/-- The setoid defined by the `SameCycle` relation. -/
def SameCycle.setoid (f : Perm α) : Setoid α where
  iseqv := SameCycle.equivalence f


@[simp]
                                                      /-
                                                        α : Type u_2
                                                        x y : α
                                                        ⊢ Iff (Equiv.Perm.SameCycle 1 x y) (Eq x y)
                                                      -/
theorem sameCycle_one : SameCycle 1 x y ↔ x = y := by simp [SameCycle]
                                                      /-
                                                        🎉 no goals
                                                      -/


@[simp]
theorem sameCycle_inv : SameCycle f⁻¹ x y ↔ SameCycle f x y :=
                                              /-
                                                α : Type u_2
                                                f : Equiv.Perm α
                                                x y : α
                                                ⊢ Iff (Exists fun b => Eq ((HPow.hPow (Inv.inv f) ((Equiv.symm (Equiv.neg Int) …
                                              -/
  (Equiv.neg _).exists_congr_left.trans <| by simp [SameCycle]
                                              /-
                                                🎉 no goals
                                              -/


alias ⟨SameCycle.of_inv, SameCycle.inv⟩ := sameCycle_inv


@[simp]
theorem sameCycle_conj : SameCycle (g * f * g⁻¹) x y ↔ SameCycle f (g⁻¹ x) (g⁻¹ y) :=
                           /-
                             α : Type u_2
                             f g : Equiv.Perm α
                             x y : α
                             i : Int
                             ⊢ Iff (Eq ((HPow.hPow (HMul.hMul (HMul.hMul g f) (Inv.inv g)) i) x) y) (Eq ((H …
                           -/
  exists_congr fun i => by simp [conj_zpow, eq_inv_iff_eq]
                           /-
                             🎉 no goals
                           -/


theorem SameCycle.conj : SameCycle f x y → SameCycle (g * f * g⁻¹) (g x) (g y) := by
  /-
    α : Type u_2
    f g : Equiv.Perm α
    x y : α
    ⊢ f.SameCycle x y → (HMul.hMul (HMul.hMul g f) (Inv.inv g)).SameCycle (g x) (g …
  -/
  simp [sameCycle_conj]
  /-
    🎉 no goals
  -/


theorem SameCycle.apply_eq_self_iff : SameCycle f x y → (f x = x ↔ f y = y) := fun ⟨i, hi⟩ => by
  rw [← hi, ← mul_apply, ← zpow_one_add, add_comm, zpow_add_one, mul_apply,
    (f ^ i).injective.eq_iff]


theorem SameCycle.eq_of_left (h : SameCycle f x y) (hx : IsFixedPt f x) : x = y :=
  let ⟨_, hn⟩ := h
  (hx.perm_zpow _).eq.symm.trans hn


theorem SameCycle.eq_of_right (h : SameCycle f x y) (hy : IsFixedPt f y) : x = y :=
  h.eq_of_left <| h.apply_eq_self_iff.2 hy


@[simp]
theorem sameCycle_apply_left : SameCycle f (f x) y ↔ SameCycle f x y :=
  (Equiv.addRight 1).exists_congr_left.trans <| by
    /-
      α : Type u_2
      f : Equiv.Perm α
      x y : α
      ⊢ Iff (Exists fun b => Eq ((HPow.hPow f ((Equiv.symm (Equiv.addRight 1)) b)) ( …
    -/
    simp [zpow_sub, SameCycle, Int.add_neg_one, Function.comp]
    /-
      🎉 no goals
    -/


@[simp]
theorem sameCycle_apply_right : SameCycle f x (f y) ↔ SameCycle f x y := by
  /-
    α : Type u_2
    f : Equiv.Perm α
    x y : α
    ⊢ Iff (f.SameCycle x (f y)) (f.SameCycle x y)
  -/
  rw [sameCycle_comm, sameCycle_apply_left, sameCycle_comm]
  /-
    🎉 no goals
  -/


@[simp]
theorem sameCycle_inv_apply_left : SameCycle f (f⁻¹ x) y ↔ SameCycle f x y := by
  /-
    α : Type u_2
    f : Equiv.Perm α
    x y : α
    ⊢ Iff (f.SameCycle ((Inv.inv f) x) y) (f.SameCycle x y)
  -/
  rw [← sameCycle_apply_left, apply_inv_self]
  /-
    🎉 no goals
  -/


@[simp]
theorem sameCycle_inv_apply_right : SameCycle f x (f⁻¹ y) ↔ SameCycle f x y := by
  /-
    α : Type u_2
    f : Equiv.Perm α
    x y : α
    ⊢ Iff (f.SameCycle x ((Inv.inv f) y)) (f.SameCycle x y)
  -/
  rw [← sameCycle_apply_right, apply_inv_self]
  /-
    🎉 no goals
  -/


@[simp]
theorem sameCycle_zpow_left {n : ℤ} : SameCycle f ((f ^ n) x) y ↔ SameCycle f x y :=
                                                         /-
                                                           α : Type u_2
                                                           f : Equiv.Perm α
                                                           x y : α
                                                           n : Int
                                                           ⊢ Iff (Exists fun b => Eq ((HPow.hPow f ((Equiv.symm (Equiv.addRight n)) b)) ( …
                                                         -/
  (Equiv.addRight (n : ℤ)).exists_congr_left.trans <| by simp [SameCycle, zpow_add]
                                                         /-
                                                           🎉 no goals
                                                         -/


@[simp]
theorem sameCycle_zpow_right {n : ℤ} : SameCycle f x ((f ^ n) y) ↔ SameCycle f x y := by
  /-
    α : Type u_2
    f : Equiv.Perm α
    x y : α
    n : Int
    ⊢ Iff (f.SameCycle x ((HPow.hPow f n) y)) (f.SameCycle x y)
  -/
  rw [sameCycle_comm, sameCycle_zpow_left, sameCycle_comm]
  /-
    🎉 no goals
  -/


@[simp]
theorem sameCycle_pow_left {n : ℕ} : SameCycle f ((f ^ n) x) y ↔ SameCycle f x y := by
  /-
    α : Type u_2
    f : Equiv.Perm α
    x y : α
    n : Nat
    ⊢ Iff (f.SameCycle ((HPow.hPow f n) x) y) (f.SameCycle x y)
  -/
  rw [← zpow_natCast, sameCycle_zpow_left]
  /-
    🎉 no goals
  -/


@[simp]
theorem sameCycle_pow_right {n : ℕ} : SameCycle f x ((f ^ n) y) ↔ SameCycle f x y := by
  /-
    α : Type u_2
    f : Equiv.Perm α
    x y : α
    n : Nat
    ⊢ Iff (f.SameCycle x ((HPow.hPow f n) y)) (f.SameCycle x y)
  -/
  rw [← zpow_natCast, sameCycle_zpow_right]
  /-
    🎉 no goals
  -/


alias ⟨SameCycle.of_apply_left, SameCycle.apply_left⟩ := sameCycle_apply_left


alias ⟨SameCycle.of_apply_right, SameCycle.apply_right⟩ := sameCycle_apply_right


alias ⟨SameCycle.of_inv_apply_left, SameCycle.inv_apply_left⟩ := sameCycle_inv_apply_left


alias ⟨SameCycle.of_inv_apply_right, SameCycle.inv_apply_right⟩ := sameCycle_inv_apply_right


alias ⟨SameCycle.of_pow_left, SameCycle.pow_left⟩ := sameCycle_pow_left


alias ⟨SameCycle.of_pow_right, SameCycle.pow_right⟩ := sameCycle_pow_right


alias ⟨SameCycle.of_zpow_left, SameCycle.zpow_left⟩ := sameCycle_zpow_left


alias ⟨SameCycle.of_zpow_right, SameCycle.zpow_right⟩ := sameCycle_zpow_right


theorem SameCycle.of_pow {n : ℕ} : SameCycle (f ^ n) x y → SameCycle f x y := fun ⟨m, h⟩ =>
             /-
               α : Type u_2
               f : Equiv.Perm α
               x y : α
               n : Nat
               x✝ : (HPow.hPow f n).SameCycle x y
               m : Int
               h : Eq ((HPow.hPow (HPow.hPow f n) m) x) y
               ⊢ Eq ((HPow.hPow f (HMul.hMul (↑n) m)) x) y
             -/
  ⟨n * m, by simp [zpow_mul, h]⟩
             /-
               🎉 no goals
             -/


theorem SameCycle.of_zpow {n : ℤ} : SameCycle (f ^ n) x y → SameCycle f x y := fun ⟨m, h⟩ =>
             /-
               α : Type u_2
               f : Equiv.Perm α
               x y : α
               n : Int
               x✝ : (HPow.hPow f n).SameCycle x y
               m : Int
               h : Eq ((HPow.hPow (HPow.hPow f n) m) x) y
               ⊢ Eq ((HPow.hPow f (HMul.hMul n m)) x) y
             -/
  ⟨n * m, by simp [zpow_mul, h]⟩
             /-
               🎉 no goals
             -/


@[simp]
theorem sameCycle_subtypePerm {h} {x y : { x // p x }} :
    (f.subtypePerm h).SameCycle x y ↔ f.SameCycle x y :=
                           /-
                             α : Type u_2
                             f : Equiv.Perm α
                             p : α → Prop
                             h : ∀ (x : α), Iff (p x) (p (f x))
                             x y : Subtype fun x => p x
                             n : Int
                             ⊢ Iff (Eq ((HPow.hPow (f.subtypePerm h) n) x) y) (Eq ((HPow.hPow f n) ↑x) ↑y)
                           -/
  exists_congr fun n => by simp [Subtype.ext_iff]
                           /-
                             🎉 no goals
                           -/


alias ⟨_, SameCycle.subtypePerm⟩ := sameCycle_subtypePerm


@[simp]
theorem sameCycle_extendDomain {p : β → Prop} [DecidablePred p] {f : α ≃ Subtype p} :
    SameCycle (g.extendDomain f) (f x) (f y) ↔ g.SameCycle x y :=
  exists_congr fun n => by
    /-
      α : Type u_2
      β : Type u_3
      g : Equiv.Perm α
      x y : α
      p : β → Prop
      inst✝ : DecidablePred p
      f : Equiv α (Subtype p)
      n : Int
      ⊢ Iff (Eq ((HPow.hPow (g.extendDomain f) n) ↑(f x)) ↑(f y)) (Eq ((HPow.hPow g  …
    -/
    rw [← extendDomain_zpow, extendDomain_apply_image, Subtype.coe_inj, f.injective.eq_iff]
    /-
      🎉 no goals
    -/


alias ⟨_, SameCycle.extendDomain⟩ := sameCycle_extendDomain


theorem SameCycle.exists_pow_eq' [Finite α] : SameCycle f x y → ∃ i < orderOf f, (f ^ i) x = y := by
  classical
    rintro ⟨k, rfl⟩
    use (k % orderOf f).natAbs
    have h₀ := Int.natCast_pos.mpr (orderOf_pos f)
    have h₁ := Int.emod_nonneg k h₀.ne'
    rw [← zpow_natCast, Int.natAbs_of_nonneg h₁, zpow_mod_orderOf]
    refine ⟨?_, by rfl⟩
    rw [← Int.ofNat_lt, Int.natAbs_of_nonneg h₁]
    exact Int.emod_lt_of_pos _ h₀


theorem SameCycle.exists_pow_eq'' [Finite α] (h : SameCycle f x y) :
    ∃ i : ℕ, 0 < i ∧ i ≤ orderOf f ∧ (f ^ i) x = y := by
  classical
    obtain ⟨_ | i, hi, rfl⟩ := h.exists_pow_eq'
    · refine ⟨orderOf f, orderOf_pos f, le_rfl, ?_⟩
      rw [pow_orderOf_eq_one, pow_zero]
    · exact ⟨i.succ, i.zero_lt_succ, hi.le, by rfl⟩


instance (f : Perm α) [DecidableRel (SameCycle f⁻¹)] :
    DecidableRel (SameCycle f) := fun x y =>
  decidable_of_iff (f⁻¹.SameCycle x y) (sameCycle_inv)


instance (f : Perm α) [DecidableRel (SameCycle f)] :
    DecidableRel (SameCycle f⁻¹) := fun x y =>
  decidable_of_iff (f.SameCycle x y) (sameCycle_inv).symm


instance (priority := 100) [DecidableEq α] : DecidableRel (SameCycle (1 : Perm α)) := fun x y =>
  decidable_of_iff (x = y) sameCycle_one.symm


instance [Fintype α] [DecidableEq α] (f : Perm α) : DecidableRel (SameCycle f) := fun x y =>
  decidable_of_iff (∃ n ∈ List.range (Fintype.card (Perm α)), (f ^ n) x = y)
    ⟨fun ⟨n, _, hn⟩ => ⟨n, hn⟩, fun ⟨i, hi⟩ => ⟨(i % orderOf f).natAbs,
      List.mem_range.2 (Int.ofNat_lt.1 <| by
        /-
          ι : Type u_1
          α : Type u_2
          β : Type u_3
          f✝ g : Equiv.Perm α
          p : α → Prop
          x✝¹ y✝ z : α
          inst✝¹ : Fintype α
          inst✝ : DecidableEq α
          f : Equiv.Perm α
          x y : α
          x✝ : f.SameCycle x y
          i : Int
          hi : Eq ((HPow.hPow f i) x) y
          ⊢ LT.lt ↑(HMod.hMod i ↑(orderOf f)).natAbs ↑(Fintype.card (Equiv.Perm α))
        -/
        rw [Int.natAbs_of_nonneg (Int.emod_nonneg _ <| Int.natCast_ne_zero.2 (orderOf_pos _).ne')]
        /-
          ι : Type u_1
          α : Type u_2
          β : Type u_3
          f✝ g : Equiv.Perm α
          p : α → Prop
          x✝¹ y✝ z : α
          inst✝¹ : Fintype α
          inst✝ : DecidableEq α
          f : Equiv.Perm α
          x y : α
          x✝ : f.SameCycle x y
          i : Int
          hi : Eq ((HPow.hPow f i) x) y
          ⊢ LT.lt (HMod.hMod i ↑(orderOf f)) ↑(Fintype.card (Equiv.Perm α))
        -/
        refine (Int.emod_lt _ <| Int.natCast_ne_zero_iff_pos.2 <| orderOf_pos _).trans_le ?_
        /-
          ι : Type u_1
          α : Type u_2
          β : Type u_3
          f✝ g : Equiv.Perm α
          p : α → Prop
          x✝¹ y✝ z : α
          inst✝¹ : Fintype α
          inst✝ : DecidableEq α
          f : Equiv.Perm α
          x y : α
          x✝ : f.SameCycle x y
          i : Int
          hi : Eq ((HPow.hPow f i) x) y
          ⊢ LE.le (abs ↑(orderOf f)) ↑(Fintype.card (Equiv.Perm α))
        -/
        simp [orderOf_le_card_univ]),
        /-
          🎉 no goals
        -/
      by
        rw [← zpow_natCast, Int.natAbs_of_nonneg (Int.emod_nonneg _ <|
          Int.natCast_ne_zero_iff_pos.2 <| orderOf_pos _), zpow_mod_orderOf, hi]⟩⟩


/-- A cycle is a non identity permutation where any two nonfixed points of the permutation are
related by repeated application of the permutation. -/
def IsCycle (f : Perm α) : Prop :=
  ∃ x, f x ≠ x ∧ ∀ ⦃y⦄, f y ≠ y → SameCycle f x y


                                                               /-
                                                                 α : Type u_2
                                                                 f : Equiv.Perm α
                                                                 h : f.IsCycle
                                                                 hf : Eq f 1
                                                                 ⊢ False
                                                               -/
theorem IsCycle.ne_one (h : IsCycle f) : f ≠ 1 := fun hf => by simp [hf, IsCycle] at h
                                                               /-
                                                                 🎉 no goals
                                                               -/


@[simp]
theorem not_isCycle_one : ¬(1 : Perm α).IsCycle := fun H => H.ne_one rfl


protected theorem IsCycle.sameCycle (hf : IsCycle f) (hx : f x ≠ x) (hy : f y ≠ y) :
    SameCycle f x y :=
  let ⟨g, hg⟩ := hf
  let ⟨a, ha⟩ := hg.2 hx
  let ⟨b, hb⟩ := hg.2 hy
             /-
               α : Type u_2
               f : Equiv.Perm α
               x y : α
               hf : f.IsCycle
               hx : Ne (f x) x
               hy : Ne (f y) y
               g : α
               hg : And (Ne (f g) g) (∀ ⦃y : α⦄, Ne (f y) y → f.SameCycle g y)
               a : Int
               ha : Eq ((HPow.hPow f a) g) x
               b : Int
               hb : Eq ((HPow.hPow f b) g) y
               ⊢ Eq ((HPow.hPow f (HSub.hSub b a)) x) y
             -/
  ⟨b - a, by rw [← ha, ← mul_apply, ← zpow_add, sub_add_cancel, hb]⟩
             /-
               🎉 no goals
             -/


theorem IsCycle.exists_zpow_eq : IsCycle f → f x ≠ x → f y ≠ y → ∃ i : ℤ, (f ^ i) x = y :=
  IsCycle.sameCycle


theorem IsCycle.inv (hf : IsCycle f) : IsCycle f⁻¹ :=
  hf.imp fun _ ⟨hx, h⟩ =>
    ⟨inv_eq_iff_eq.not.2 hx.symm, fun _ hy => (h <| inv_eq_iff_eq.not.2 hy.symm).inv⟩


@[simp]
theorem isCycle_inv : IsCycle f⁻¹ ↔ IsCycle f :=
  ⟨fun h => h.inv, IsCycle.inv⟩


theorem IsCycle.conj : IsCycle f → IsCycle (g * f * g⁻¹) := by
  /-
    α : Type u_2
    f g : Equiv.Perm α
    ⊢ f.IsCycle → (HMul.hMul (HMul.hMul g f) (Inv.inv g)).IsCycle
  -/
  rintro ⟨x, hx, h⟩
  /-
    case intro.intro
    α : Type u_2
    f g : Equiv.Perm α
    x : α
    hx : Ne (f x) x
    h : ∀ ⦃y : α⦄, Ne (f y) y → f.SameCycle x y
    ⊢ (HMul.hMul (HMul.hMul g f) (Inv.inv g)).IsCycle
  -/
  refine ⟨g x, by simp [coe_mul, inv_apply_self, hx], fun y hy => ?_⟩
  /-
    case intro.intro
    α : Type u_2
    f g : Equiv.Perm α
    x : α
    hx : Ne (f x) x
    h : ∀ ⦃y : α⦄, Ne (f y) y → f.SameCycle x y
    y : α
    hy : Ne ((HMul.hMul (HMul.hMul g f) (Inv.inv g)) y) y
    ⊢ (HMul.hMul (HMul.hMul g f) (Inv.inv g)).SameCycle (g x) y
  -/
  rw [← apply_inv_self g y]
  /-
    case intro.intro
    α : Type u_2
    f g : Equiv.Perm α
    x : α
    hx : Ne (f x) x
    h : ∀ ⦃y : α⦄, Ne (f y) y → f.SameCycle x y
    y : α
    hy : Ne ((HMul.hMul (HMul.hMul g f) (Inv.inv g)) y) y
    ⊢ (HMul.hMul (HMul.hMul g f) (Inv.inv g)).SameCycle (g x) (g ((Inv.inv g) y))
  -/
  exact (h <| eq_inv_iff_eq.not.2 hy).conj
  /-
    🎉 no goals
  -/


protected theorem IsCycle.extendDomain {p : β → Prop} [DecidablePred p] (f : α ≃ Subtype p) :
    IsCycle g → IsCycle (g.extendDomain f) := by
  /-
    α : Type u_2
    β : Type u_3
    g : Equiv.Perm α
    p : β → Prop
    inst✝ : DecidablePred p
    f : Equiv α (Subtype p)
    ⊢ g.IsCycle → (g.extendDomain f).IsCycle
  -/
  rintro ⟨a, ha, ha'⟩
  /-
    case intro.intro
    α : Type u_2
    β : Type u_3
    g : Equiv.Perm α
    p : β → Prop
    inst✝ : DecidablePred p
    f : Equiv α (Subtype p)
    a : α
    ha : Ne (g a) a
    ha' : ∀ ⦃y : α⦄, Ne (g y) y → g.SameCycle a y
    ⊢ (g.extendDomain f).IsCycle
  -/
  refine ⟨f a, ?_, fun b hb => ?_⟩
    /-
      case intro.intro.refine_1
      α : Type u_2
      β : Type u_3
      g : Equiv.Perm α
      p : β → Prop
      inst✝ : DecidablePred p
      f : Equiv α (Subtype p)
      a : α
      ha : Ne (g a) a
      ha' : ∀ ⦃y : α⦄, Ne (g y) y → g.SameCycle a y
      ⊢ Ne ((g.extendDomain f) ↑(f a)) ↑(f a)
    -/
  · rw [extendDomain_apply_image]
    /-
      case intro.intro.refine_1
      α : Type u_2
      β : Type u_3
      g : Equiv.Perm α
      p : β → Prop
      inst✝ : DecidablePred p
      f : Equiv α (Subtype p)
      a : α
      ha : Ne (g a) a
      ha' : ∀ ⦃y : α⦄, Ne (g y) y → g.SameCycle a y
      ⊢ Ne ↑(f (g a)) ↑(f a)
    -/
    exact Subtype.coe_injective.ne (f.injective.ne ha)
    /-
      🎉 no goals
    -/
  have h : b = f (f.symm ⟨b, of_not_not <| hb ∘ extendDomain_apply_not_subtype _ _⟩) := by
    rw [apply_symm_apply, Subtype.coe_mk]
  /-
    case intro.intro.refine_2
    α : Type u_2
    β : Type u_3
    g : Equiv.Perm α
    p : β → Prop
    inst✝ : DecidablePred p
    f : Equiv α (Subtype p)
    a : α
    ha : Ne (g a) a
    ha' : ∀ ⦃y : α⦄, Ne (g y) y → g.SameCycle a y
    b : β
    hb : Ne ((g.extendDomain f) b) b
    h : Eq b ↑(f (f.symm ⟨b, ⋯⟩))
    ⊢ (g.extendDomain f).SameCycle (↑(f a)) b
  -/
  rw [h] at hb ⊢
  /-
    case intro.intro.refine_2
    α : Type u_2
    β : Type u_3
    g : Equiv.Perm α
    p : β → Prop
    inst✝ : DecidablePred p
    f : Equiv α (Subtype p)
    a : α
    ha : Ne (g a) a
    ha' : ∀ ⦃y : α⦄, Ne (g y) y → g.SameCycle a y
    b : β
    hb✝ : Ne ((g.extendDomain f) b) b
    hb : Ne ((g.extendDomain f) ↑(f (f.symm ⟨b, ⋯⟩))) ↑(f (f.symm ⟨b, ⋯⟩))
    h : Eq b ↑(f (f.symm ⟨b, ⋯⟩))
    ⊢ (g.extendDomain f).SameCycle ↑(f a) ↑(f (f.symm ⟨b, ⋯⟩))
  -/
  simp only [extendDomain_apply_image, Subtype.coe_injective.ne_iff, f.injective.ne_iff] at hb
  /-
    case intro.intro.refine_2
    α : Type u_2
    β : Type u_3
    g : Equiv.Perm α
    p : β → Prop
    inst✝ : DecidablePred p
    f : Equiv α (Subtype p)
    a : α
    ha : Ne (g a) a
    ha' : ∀ ⦃y : α⦄, Ne (g y) y → g.SameCycle a y
    b : β
    hb✝ : Ne ((g.extendDomain f) b) b
    h : Eq b ↑(f (f.symm ⟨b, ⋯⟩))
    hb : Ne (g (f.symm ⟨b, ⋯⟩)) (f.symm ⟨b, ⋯⟩)
    ⊢ (g.extendDomain f).SameCycle ↑(f a) ↑(f (f.symm ⟨b, ⋯⟩))
  -/
  exact (ha' hb).extendDomain
  /-
    🎉 no goals
  -/


theorem isCycle_iff_sameCycle (hx : f x ≠ x) : IsCycle f ↔ ∀ {y}, SameCycle f x y ↔ f y ≠ y :=
  ⟨fun hf y =>
    ⟨fun ⟨i, hi⟩ hy =>
      hx <| by
        /-
          α : Type u_2
          f : Equiv.Perm α
          x : α
          hx : Ne (f x) x
          hf : f.IsCycle
          y : α
          x✝ : f.SameCycle x y
          hy : Eq (f y) y
          i : Int
          hi : Eq ((HPow.hPow f i) x) y
          ⊢ Eq (f x) x
        -/
        rw [← zpow_apply_eq_self_of_apply_eq_self hy i, (f ^ i).injective.eq_iff] at hi
        /-
          α : Type u_2
          f : Equiv.Perm α
          x : α
          hx : Ne (f x) x
          hf : f.IsCycle
          y : α
          x✝ : f.SameCycle x y
          hy : Eq (f y) y
          i : Int
          hi : Eq x y
          ⊢ Eq (f x) x
        -/
        rw [hi, hy],
        /-
          🎉 no goals
        -/
      hf.exists_zpow_eq hx⟩,
    fun h => ⟨x, hx, fun _ hy => h.2 hy⟩⟩


theorem IsCycle.exists_pow_eq (hf : IsCycle f) (hx : f x ≠ x) (hy : f y ≠ y) :
    ∃ i : ℕ, (f ^ i) x = y := by
  /-
    α : Type u_2
    f : Equiv.Perm α
    x y : α
    inst✝ : Finite α
    hf : f.IsCycle
    hx : Ne (f x) x
    hy : Ne (f y) y
    ⊢ Exists fun i => Eq ((HPow.hPow f i) x) y
  -/
  let ⟨n, hn⟩ := hf.exists_zpow_eq hx hy
  classical exact
      ⟨(n % orderOf f).toNat, by
        {have := n.emod_nonneg (Int.natCast_ne_zero.mpr (ne_of_gt (orderOf_pos f)))
         rwa [← zpow_natCast, Int.toNat_of_nonneg this, zpow_mod_orderOf]}⟩


theorem isCycle_swap (hxy : x ≠ y) : IsCycle (swap x y) :=
         /-
           α : Type u_2
           x y : α
           inst✝ : DecidableEq α
           hxy : Ne x y
           ⊢ Ne ((Equiv.swap x y) y) y
         -/
  ⟨y, by rwa [swap_apply_right], fun a (ha : ite (a = x) y (ite (a = y) x a) ≠ a) =>
         /-
           🎉 no goals
         -/
    if hya : y = a then ⟨0, hya⟩
    else
      ⟨1, by
        /-
          α : Type u_2
          x y : α
          inst✝ : DecidableEq α
          hxy : Ne x y
          a : α
          ha : Ne (ite (Eq a x) y (ite (Eq a y) x a)) a
          hya : Not (Eq y a)
          ⊢ Eq ((HPow.hPow (Equiv.swap x y) 1) y) a
        -/
        rw [zpow_one, swap_apply_def]
        /-
          α : Type u_2
          x y : α
          inst✝ : DecidableEq α
          hxy : Ne x y
          a : α
          ha : Ne (ite (Eq a x) y (ite (Eq a y) x a)) a
          hya : Not (Eq y a)
          ⊢ Eq (ite (Eq y x) y (ite (Eq y y) x y)) a
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
                           /-
                             🎉 no goals
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
                           /-
                             🎉 no goals
                           -/
        split_ifs at * <;> tauto⟩⟩
                           /-
                             🎉 no goals
                           -/


protected theorem IsSwap.isCycle : IsSwap f → IsCycle f := by
  /-
    α : Type u_2
    f : Equiv.Perm α
    inst✝ : DecidableEq α
    ⊢ f.IsSwap → f.IsCycle
  -/
  rintro ⟨x, y, hxy, rfl⟩
  /-
    case intro.intro.intro
    α : Type u_2
    inst✝ : DecidableEq α
    x y : α
    hxy : Ne x y
    ⊢ (Equiv.swap x y).IsCycle
  -/
  exact isCycle_swap hxy
  /-
    🎉 no goals
  -/


theorem IsCycle.two_le_card_support (h : IsCycle f) : 2 ≤ #f.support :=
  two_le_card_support_of_ne_one h.ne_one


/-- The subgroup generated by a cycle is in bijection with its support -/
noncomputable def IsCycle.zpowersEquivSupport {σ : Perm α} (hσ : IsCycle σ) :
    (Subgroup.zpowers σ) ≃ σ.support :=
  Equiv.ofBijective
    (fun (τ : ↥ ((Subgroup.zpowers σ) : Set (Perm α))) =>
      ⟨(τ : Perm α) (Classical.choose hσ), by
        /-
          ι : Type u_1
          α : Type u_2
          β : Type u_3
          f g : Equiv.Perm α
          x y : α
          inst✝¹ : DecidableEq α
          inst✝ : Fintype α
          σ : Equiv.Perm α
          hσ : σ.IsCycle
          τ : ↑↑(Subgroup.zpowers σ)
          ⊢ Membership.mem σ.support (↑τ (Classical.choose hσ))
        -/
        obtain ⟨τ, n, rfl⟩ := τ
        /-
          case mk.intro
          ι : Type u_1
          α : Type u_2
          β : Type u_3
          f g : Equiv.Perm α
          x y : α
          inst✝¹ : DecidableEq α
          inst✝ : Fintype α
          σ : Equiv.Perm α
          hσ : σ.IsCycle
          n : Int
          ⊢ Membership.mem σ.support (↑⟨(fun x => HPow.hPow σ x) n, ⋯⟩ (Classical.choose …
        -/
        erw [Finset.mem_coe, Subtype.coe_mk, zpow_apply_mem_support, mem_support]
        /-
          case mk.intro
          ι : Type u_1
          α : Type u_2
          β : Type u_3
          f g : Equiv.Perm α
          x y : α
          inst✝¹ : DecidableEq α
          inst✝ : Fintype α
          σ : Equiv.Perm α
          hσ : σ.IsCycle
          n : Int
          ⊢ Ne (σ (Classical.choose hσ)) (Classical.choose hσ)
        -/
        exact (Classical.choose_spec hσ).1⟩)
        /-
          🎉 no goals
        -/
    (by
      /-
        ι : Type u_1
        α : Type u_2
        β : Type u_3
        f g : Equiv.Perm α
        x y : α
        inst✝¹ : DecidableEq α
        inst✝ : Fintype α
        σ : Equiv.Perm α
        hσ : σ.IsCycle
        ⊢ Function.Bijective fun τ => ⟨↑τ (Classical.choose hσ), ⋯⟩
      -/
      constructor
        /-
          case left
          ι : Type u_1
          α : Type u_2
          β : Type u_3
          f g : Equiv.Perm α
          x y : α
          inst✝¹ : DecidableEq α
          inst✝ : Fintype α
          σ : Equiv.Perm α
          hσ : σ.IsCycle
          ⊢ Function.Injective fun τ => ⟨↑τ (Classical.choose hσ), ⋯⟩
        -/
      · rintro ⟨a, m, rfl⟩ ⟨b, n, rfl⟩ h
        /-
          case left.mk.intro.mk.intro
          ι : Type u_1
          α : Type u_2
          β : Type u_3
          f g : Equiv.Perm α
          x y : α
          inst✝¹ : DecidableEq α
          inst✝ : Fintype α
          σ : Equiv.Perm α
          hσ : σ.IsCycle
          m n : Int
          h : Eq ((fun τ => ⟨↑τ (Classical.choose hσ), ⋯⟩) ⟨(fun x => HPow.hPow σ x) m,  …
          ⊢ Eq ⟨(fun x => HPow.hPow σ x) m, ⋯⟩ ⟨(fun x => HPow.hPow σ x) n, ⋯⟩
        -/
        ext y
        /-
          case left.mk.intro.mk.intro.a.H
          ι : Type u_1
          α : Type u_2
          β : Type u_3
          f g : Equiv.Perm α
          x y✝ : α
          inst✝¹ : DecidableEq α
          inst✝ : Fintype α
          σ : Equiv.Perm α
          hσ : σ.IsCycle
          m n : Int
          h : Eq ((fun τ => ⟨↑τ (Classical.choose hσ), ⋯⟩) ⟨(fun x => HPow.hPow σ x) m,  …
          y : α
          ⊢ Eq (↑⟨(fun x => HPow.hPow σ x) m, ⋯⟩ y) (↑⟨(fun x => HPow.hPow σ x) n, ⋯⟩ y)
        -/
        by_cases hy : σ y = y
          /-
            case pos
            ι : Type u_1
            α : Type u_2
            β : Type u_3
            f g : Equiv.Perm α
            x y✝ : α
            inst✝¹ : DecidableEq α
            inst✝ : Fintype α
            σ : Equiv.Perm α
            hσ : σ.IsCycle
            m n : Int
            h : Eq ((fun τ => ⟨↑τ (Classical.choose hσ), ⋯⟩) ⟨(fun x => HPow.hPow σ x) m,  …
            y : α
            hy : Eq (σ y) y
            ⊢ Eq (↑⟨(fun x => HPow.hPow σ x) m, ⋯⟩ y) (↑⟨(fun x => HPow.hPow σ x) n, ⋯⟩ y)
          -/
        · simp_rw [zpow_apply_eq_self_of_apply_eq_self hy]
          /-
            🎉 no goals
          -/
          /-
            case neg
            ι : Type u_1
            α : Type u_2
            β : Type u_3
            f g : Equiv.Perm α
            x y✝ : α
            inst✝¹ : DecidableEq α
            inst✝ : Fintype α
            σ : Equiv.Perm α
            hσ : σ.IsCycle
            m n : Int
            h : Eq ((fun τ => ⟨↑τ (Classical.choose hσ), ⋯⟩) ⟨(fun x => HPow.hPow σ x) m,  …
            y : α
            hy : Not (Eq (σ y) y)
            ⊢ Eq (↑⟨(fun x => HPow.hPow σ x) m, ⋯⟩ y) (↑⟨(fun x => HPow.hPow σ x) n, ⋯⟩ y)
          -/
        · obtain ⟨i, rfl⟩ := (Classical.choose_spec hσ).2 hy
          /-
            case neg.intro
            ι : Type u_1
            α : Type u_2
            β : Type u_3
            f g : Equiv.Perm α
            x y : α
            inst✝¹ : DecidableEq α
            inst✝ : Fintype α
            σ : Equiv.Perm α
            hσ : σ.IsCycle
            m n : Int
            h : Eq ((fun τ => ⟨↑τ (Classical.choose hσ), ⋯⟩) ⟨(fun x => HPow.hPow σ x) m,  …
            i : Int
            hy : Not (Eq (σ ((HPow.hPow σ i) (Classical.choose hσ))) ((HPow.hPow σ i) (Cla …
            ⊢ Eq (↑⟨(fun x => HPow.hPow σ x) m, ⋯⟩ ((HPow.hPow σ i) (Classical.choose hσ)) …
          -/
          rw [Subtype.coe_mk, Subtype.coe_mk, zpow_apply_comm σ m i, zpow_apply_comm σ n i]
          /-
            case neg.intro
            ι : Type u_1
            α : Type u_2
            β : Type u_3
            f g : Equiv.Perm α
            x y : α
            inst✝¹ : DecidableEq α
            inst✝ : Fintype α
            σ : Equiv.Perm α
            hσ : σ.IsCycle
            m n : Int
            h : Eq ((fun τ => ⟨↑τ (Classical.choose hσ), ⋯⟩) ⟨(fun x => HPow.hPow σ x) m,  …
            i : Int
            hy : Not (Eq (σ ((HPow.hPow σ i) (Classical.choose hσ))) ((HPow.hPow σ i) (Cla …
            ⊢ Eq ((HPow.hPow σ i) ((HPow.hPow σ m) (Classical.choose hσ))) ((HPow.hPow σ i …
          -/
          exact congr_arg _ (Subtype.ext_iff.mp h)
          /-
            🎉 no goals
          -/
        /-
          case right
          ι : Type u_1
          α : Type u_2
          β : Type u_3
          f g : Equiv.Perm α
          x y : α
          inst✝¹ : DecidableEq α
          inst✝ : Fintype α
          σ : Equiv.Perm α
          hσ : σ.IsCycle
          ⊢ Function.Surjective fun τ => ⟨↑τ (Classical.choose hσ), ⋯⟩
        -/
      · rintro ⟨y, hy⟩
        /-
          case right.mk
          ι : Type u_1
          α : Type u_2
          β : Type u_3
          f g : Equiv.Perm α
          x y✝ : α
          inst✝¹ : DecidableEq α
          inst✝ : Fintype α
          σ : Equiv.Perm α
          hσ : σ.IsCycle
          y : α
          hy : Membership.mem σ.support y
          ⊢ Exists fun a => Eq ((fun τ => ⟨↑τ (Classical.choose hσ), ⋯⟩) a) ⟨y, hy⟩
        -/
        erw [Finset.mem_coe, mem_support] at hy
        /-
          case right.mk
          ι : Type u_1
          α : Type u_2
          β : Type u_3
          f g : Equiv.Perm α
          x y✝ : α
          inst✝¹ : DecidableEq α
          inst✝ : Fintype α
          σ : Equiv.Perm α
          hσ : σ.IsCycle
          y : α
          hy✝ : Membership.mem σ.support y
          hy : Ne (σ y) y
          ⊢ Exists fun a => Eq ((fun τ => ⟨↑τ (Classical.choose hσ), ⋯⟩) a) ⟨y, hy✝⟩
        -/
        obtain ⟨n, rfl⟩ := (Classical.choose_spec hσ).2 hy
        /-
          case right.mk.intro
          ι : Type u_1
          α : Type u_2
          β : Type u_3
          f g : Equiv.Perm α
          x y : α
          inst✝¹ : DecidableEq α
          inst✝ : Fintype α
          σ : Equiv.Perm α
          hσ : σ.IsCycle
          n : Int
          hy✝ : Membership.mem σ.support ((HPow.hPow σ n) (Classical.choose hσ))
          hy : Ne (σ ((HPow.hPow σ n) (Classical.choose hσ))) ((HPow.hPow σ n) (Classica …
          ⊢ Exists fun a => Eq ((fun τ => ⟨↑τ (Classical.choose hσ), ⋯⟩) a) ⟨(HPow.hPow  …
        -/
        exact ⟨⟨σ ^ n, n, rfl⟩, rfl⟩)
        /-
          🎉 no goals
        -/


@[simp]
theorem IsCycle.zpowersEquivSupport_apply {σ : Perm α} (hσ : IsCycle σ) {n : ℕ} :
    hσ.zpowersEquivSupport ⟨σ ^ n, n, rfl⟩ =
      ⟨(σ ^ n) (Classical.choose hσ),
        pow_apply_mem_support.2 (mem_support.2 (Classical.choose_spec hσ).1)⟩ :=
  rfl


@[simp]
theorem IsCycle.zpowersEquivSupport_symm_apply {σ : Perm α} (hσ : IsCycle σ) (n : ℕ) :
    hσ.zpowersEquivSupport.symm
        ⟨(σ ^ n) (Classical.choose hσ),
          pow_apply_mem_support.2 (mem_support.2 (Classical.choose_spec hσ).1)⟩ =
      ⟨σ ^ n, n, rfl⟩ :=
  (Equiv.symm_apply_eq _).2 hσ.zpowersEquivSupport_apply


protected theorem IsCycle.orderOf (hf : IsCycle f) : orderOf f = #f.support := by
  /-
    α : Type u_2
    f : Equiv.Perm α
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    hf : f.IsCycle
    ⊢ Eq (orderOf f) f.support.card
  -/
  rw [← Fintype.card_zpowers, ← Fintype.card_coe]
  /-
    α : Type u_2
    f : Equiv.Perm α
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    hf : f.IsCycle
    ⊢ Eq (Fintype.card (Subtype fun x => Membership.mem (Subgroup.zpowers f) x)) ( …
  -/
  convert Fintype.card_congr (IsCycle.zpowersEquivSupport hf)
  /-
    🎉 no goals
  -/


theorem isCycle_swap_mul_aux₁ {α : Type*} [DecidableEq α] :
    ∀ (n : ℕ) {b x : α} {f : Perm α} (_ : (swap x (f x) * f) b ≠ b) (_ : (f ^ n) (f x) = b),
      ∃ i : ℤ, ((swap x (f x) * f) ^ i) (f x) = b := by
  /-
    α : Type u_4
    inst✝ : DecidableEq α
    ⊢ ∀ (n : Nat) {b x : α} {f : Equiv.Perm α}, Ne ((HMul.hMul (Equiv.swap x (f x) …
  -/
  intro n
  induction n with
  | zero => exact fun _ h => ⟨0, h⟩
  | succ n hn =>
    intro b x f hb h
    exact if hfbx : f x = b then ⟨0, hfbx⟩
      else
        have : f b ≠ b ∧ b ≠ x := ne_and_ne_of_swap_mul_apply_ne_self hb
        have hb' : (swap x (f x) * f) (f⁻¹ b) ≠ f⁻¹ b := by
          rw [mul_apply, apply_inv_self, swap_apply_of_ne_of_ne this.2 (Ne.symm hfbx), Ne, ←
            f.injective.eq_iff, apply_inv_self]
          exact this.1
        let ⟨i, hi⟩ := hn hb' (f.injective <| by
          rw [apply_inv_self]; rwa [pow_succ', mul_apply] at h)
        ⟨i + 1, by
          rw [add_comm, zpow_add, mul_apply, hi, zpow_one, mul_apply, apply_inv_self,
            swap_apply_of_ne_of_ne (ne_and_ne_of_swap_mul_apply_ne_self hb).2 (Ne.symm hfbx)]⟩


theorem isCycle_swap_mul_aux₂ {α : Type*} [DecidableEq α] :
    ∀ (n : ℤ) {b x : α} {f : Perm α} (_ : (swap x (f x) * f) b ≠ b) (_ : (f ^ n) (f x) = b),
      ∃ i : ℤ, ((swap x (f x) * f) ^ i) (f x) = b := by
  /-
    α : Type u_4
    inst✝ : DecidableEq α
    ⊢ ∀ (n : Int) {b x : α} {f : Equiv.Perm α}, Ne ((HMul.hMul (Equiv.swap x (f x) …
  -/
  intro n
  induction n with
  | ofNat n => exact isCycle_swap_mul_aux₁ n
  | negSucc n =>
    intro b x f hb h
    exact if hfbx' : f x = b then ⟨0, hfbx'⟩
      else
        have : f b ≠ b ∧ b ≠ x := ne_and_ne_of_swap_mul_apply_ne_self hb
        have hb : (swap x (f⁻¹ x) * f⁻¹) (f⁻¹ b) ≠ f⁻¹ b := by
          rw [mul_apply, swap_apply_def]
          split_ifs <;>
            simp only [inv_eq_iff_eq, Perm.mul_apply, zpow_negSucc, Ne, Perm.apply_inv_self] at *
              <;> tauto
        let ⟨i, hi⟩ :=
          isCycle_swap_mul_aux₁ n hb
            (show (f⁻¹ ^ n) (f⁻¹ x) = f⁻¹ b by
              rw [← zpow_natCast, ← h, ← mul_apply, ← mul_apply, ← mul_apply, zpow_negSucc,
                ← inv_pow, pow_succ, mul_assoc, mul_assoc, inv_mul_cancel, mul_one, zpow_natCast,
                ← pow_succ', ← pow_succ])
        have h : (swap x (f⁻¹ x) * f⁻¹) (f x) = f⁻¹ x := by
          rw [mul_apply, inv_apply_self, swap_apply_left]
        ⟨-i, by
          rw [← add_sub_cancel_right i 1, neg_sub, sub_eq_add_neg, zpow_add, zpow_one, zpow_neg,
            ← inv_zpow, mul_inv_rev, swap_inv, mul_swap_eq_swap_mul, inv_apply_self, swap_comm _ x,
            zpow_add, zpow_one, mul_apply, mul_apply (_ ^ i), h, hi, mul_apply, apply_inv_self,
            swap_apply_of_ne_of_ne this.2 (Ne.symm hfbx')]⟩


theorem IsCycle.eq_swap_of_apply_apply_eq_self {α : Type*} [DecidableEq α] {f : Perm α}
    (hf : IsCycle f) {x : α} (hfx : f x ≠ x) (hffx : f (f x) = x) : f = swap x (f x) :=
  Equiv.ext fun y =>
    let ⟨z, hz⟩ := hf
    let ⟨i, hi⟩ := hz.2 hfx
                           /-
                             α : Type u_4
                             inst✝ : DecidableEq α
                             f : Equiv.Perm α
                             hf : f.IsCycle
                             x : α
                             hfx : Ne (f x) x
                             hffx : Eq (f (f x)) x
                             y z : α
                             hz : And (Ne (f z) z) (∀ ⦃y : α⦄, Ne (f y) y → f.SameCycle z y)
                             i : Int
                             hi : Eq ((HPow.hPow f i) z) x
                             hyx : Eq y x
                             ⊢ Eq (f y) ((Equiv.swap x (f x)) y)
                           -/
    if hyx : y = x then by simp [hyx]
                           /-
                             🎉 no goals
                           -/
    else
                                /-
                                  α : Type u_4
                                  inst✝ : DecidableEq α
                                  f : Equiv.Perm α
                                  hf : f.IsCycle
                                  x : α
                                  hfx : Ne (f x) x
                                  hffx : Eq (f (f x)) x
                                  y z : α
                                  hz : And (Ne (f z) z) (∀ ⦃y : α⦄, Ne (f y) y → f.SameCycle z y)
                                  i : Int
                                  hi : Eq ((HPow.hPow f i) z) x
                                  hyx : Not (Eq y x)
                                  hfyx : Eq y (f x)
                                  ⊢ Eq (f y) ((Equiv.swap x (f x)) y)
                                -/
      if hfyx : y = f x then by simp [hfyx, hffx]
                                /-
                                  🎉 no goals
                                -/
      else by
        /-
          α : Type u_4
          inst✝ : DecidableEq α
          f : Equiv.Perm α
          hf : f.IsCycle
          x : α
          hfx : Ne (f x) x
          hffx : Eq (f (f x)) x
          y z : α
          hz : And (Ne (f z) z) (∀ ⦃y : α⦄, Ne (f y) y → f.SameCycle z y)
          i : Int
          hi : Eq ((HPow.hPow f i) z) x
          hyx : Not (Eq y x)
          hfyx : Not (Eq y (f x))
          ⊢ Eq (f y) ((Equiv.swap x (f x)) y)
        -/
        rw [swap_apply_of_ne_of_ne hyx hfyx]
        /-
          α : Type u_4
          inst✝ : DecidableEq α
          f : Equiv.Perm α
          hf : f.IsCycle
          x : α
          hfx : Ne (f x) x
          hffx : Eq (f (f x)) x
          y z : α
          hz : And (Ne (f z) z) (∀ ⦃y : α⦄, Ne (f y) y → f.SameCycle z y)
          i : Int
          hi : Eq ((HPow.hPow f i) z) x
          hyx : Not (Eq y x)
          hfyx : Not (Eq y (f x))
          ⊢ Eq (f y) y
        -/
        refine by_contradiction fun hy => ?_
        /-
          α : Type u_4
          inst✝ : DecidableEq α
          f : Equiv.Perm α
          hf : f.IsCycle
          x : α
          hfx : Ne (f x) x
          hffx : Eq (f (f x)) x
          y z : α
          hz : And (Ne (f z) z) (∀ ⦃y : α⦄, Ne (f y) y → f.SameCycle z y)
          i : Int
          hi : Eq ((HPow.hPow f i) z) x
          hyx : Not (Eq y x)
          hfyx : Not (Eq y (f x))
          hy : Not (Eq (f y) y)
          ⊢ False
        -/
        cases' hz.2 hy with j hj
        /-
          case intro
          α : Type u_4
          inst✝ : DecidableEq α
          f : Equiv.Perm α
          hf : f.IsCycle
          x : α
          hfx : Ne (f x) x
          hffx : Eq (f (f x)) x
          y z : α
          hz : And (Ne (f z) z) (∀ ⦃y : α⦄, Ne (f y) y → f.SameCycle z y)
          i : Int
          hi : Eq ((HPow.hPow f i) z) x
          hyx : Not (Eq y x)
          hfyx : Not (Eq y (f x))
          hy : Not (Eq (f y) y)
          j : Int
          hj : Eq ((HPow.hPow f j) z) y
          ⊢ False
        -/
        rw [← sub_add_cancel j i, zpow_add, mul_apply, hi] at hj
        /-
          case intro
          α : Type u_4
          inst✝ : DecidableEq α
          f : Equiv.Perm α
          hf : f.IsCycle
          x : α
          hfx : Ne (f x) x
          hffx : Eq (f (f x)) x
          y z : α
          hz : And (Ne (f z) z) (∀ ⦃y : α⦄, Ne (f y) y → f.SameCycle z y)
          i : Int
          hi : Eq ((HPow.hPow f i) z) x
          hyx : Not (Eq y x)
          hfyx : Not (Eq y (f x))
          hy : Not (Eq (f y) y)
          j : Int
          hj : Eq ((HPow.hPow f (HSub.hSub j i)) x) y
          ⊢ False
        -/
        cases' zpow_apply_eq_of_apply_apply_eq_self hffx (j - i) with hji hji
          /-
            case intro.inl
            α : Type u_4
            inst✝ : DecidableEq α
            f : Equiv.Perm α
            hf : f.IsCycle
            x : α
            hfx : Ne (f x) x
            hffx : Eq (f (f x)) x
            y z : α
            hz : And (Ne (f z) z) (∀ ⦃y : α⦄, Ne (f y) y → f.SameCycle z y)
            i : Int
            hi : Eq ((HPow.hPow f i) z) x
            hyx : Not (Eq y x)
            hfyx : Not (Eq y (f x))
            hy : Not (Eq (f y) y)
            j : Int
            hj : Eq ((HPow.hPow f (HSub.hSub j i)) x) y
            hji : Eq ((HPow.hPow f (HSub.hSub j i)) x) x
            ⊢ False
          -/
        · rw [← hj, hji] at hyx
          /-
            case intro.inl
            α : Type u_4
            inst✝ : DecidableEq α
            f : Equiv.Perm α
            hf : f.IsCycle
            x : α
            hfx : Ne (f x) x
            hffx : Eq (f (f x)) x
            y z : α
            hz : And (Ne (f z) z) (∀ ⦃y : α⦄, Ne (f y) y → f.SameCycle z y)
            i : Int
            hi : Eq ((HPow.hPow f i) z) x
            hfyx : Not (Eq y (f x))
            hy : Not (Eq (f y) y)
            j : Int
            hyx : Not (Eq x x)
            hj : Eq ((HPow.hPow f (HSub.hSub j i)) x) y
            hji : Eq ((HPow.hPow f (HSub.hSub j i)) x) x
            ⊢ False
          -/
          tauto
          /-
            🎉 no goals
          -/
          /-
            case intro.inr
            α : Type u_4
            inst✝ : DecidableEq α
            f : Equiv.Perm α
            hf : f.IsCycle
            x : α
            hfx : Ne (f x) x
            hffx : Eq (f (f x)) x
            y z : α
            hz : And (Ne (f z) z) (∀ ⦃y : α⦄, Ne (f y) y → f.SameCycle z y)
            i : Int
            hi : Eq ((HPow.hPow f i) z) x
            hyx : Not (Eq y x)
            hfyx : Not (Eq y (f x))
            hy : Not (Eq (f y) y)
            j : Int
            hj : Eq ((HPow.hPow f (HSub.hSub j i)) x) y
            hji : Eq ((HPow.hPow f (HSub.hSub j i)) x) (f x)
            ⊢ False
          -/
        · rw [← hj, hji] at hfyx
          /-
            case intro.inr
            α : Type u_4
            inst✝ : DecidableEq α
            f : Equiv.Perm α
            hf : f.IsCycle
            x : α
            hfx : Ne (f x) x
            hffx : Eq (f (f x)) x
            y z : α
            hz : And (Ne (f z) z) (∀ ⦃y : α⦄, Ne (f y) y → f.SameCycle z y)
            i : Int
            hi : Eq ((HPow.hPow f i) z) x
            hyx : Not (Eq y x)
            hy : Not (Eq (f y) y)
            j : Int
            hfyx : Not (Eq (f x) (f x))
            hj : Eq ((HPow.hPow f (HSub.hSub j i)) x) y
            hji : Eq ((HPow.hPow f (HSub.hSub j i)) x) (f x)
            ⊢ False
          -/
          tauto
          /-
            🎉 no goals
          -/


theorem IsCycle.swap_mul {α : Type*} [DecidableEq α] {f : Perm α} (hf : IsCycle f) {x : α}
    (hx : f x ≠ x) (hffx : f (f x) ≠ x) : IsCycle (swap x (f x) * f) :=
           /-
             α : Type u_4
             inst✝ : DecidableEq α
             f : Equiv.Perm α
             hf : f.IsCycle
             x : α
             hx : Ne (f x) x
             hffx : Ne (f (f x)) x
             ⊢ Ne ((HMul.hMul (Equiv.swap x (f x)) f) (f x)) (f x)
           -/
  ⟨f x, by simp [swap_apply_def, mul_apply, if_neg hffx, f.injective.eq_iff, if_neg hx, hx],
           /-
             🎉 no goals
           -/
    fun y hy =>
    let ⟨i, hi⟩ := hf.exists_zpow_eq hx (ne_and_ne_of_swap_mul_apply_ne_self hy).1
    -- Porting note: Needed to add Perm α typehint, otherwise does not know how to coerce to fun
    have hi : (f ^ (i - 1) : Perm α) (f x) = y :=
      calc
                                                                                    /-
                                                                                      α : Type u_4
                                                                                      inst✝ : DecidableEq α
                                                                                      f : Equiv.Perm α
                                                                                      hf : f.IsCycle
                                                                                      x : α
                                                                                      hx : Ne (f x) x
                                                                                      hffx : Ne (f (f x)) x
                                                                                      y : α
                                                                                      hy : Ne ((HMul.hMul (Equiv.swap x (f x)) f) y) y
                                                                                      i : Int
                                                                                      hi : Eq ((HPow.hPow f i) x) y
                                                                                      ⊢ Eq ((HPow.hPow f (HSub.hSub i 1)) (f x)) ((HMul.hMul (HPow.hPow f (HSub.hSub …
                                                                                    -/
        (f ^ (i - 1) : Perm α) (f x) = (f ^ (i - 1) * f ^ (1 : ℤ) : Perm α) x := by simp
                                                                                    /-
                                                                                      🎉 no goals
                                                                                    -/
                    /-
                      α : Type u_4
                      inst✝ : DecidableEq α
                      f : Equiv.Perm α
                      hf : f.IsCycle
                      x : α
                      hx : Ne (f x) x
                      hffx : Ne (f (f x)) x
                      y : α
                      hy : Ne ((HMul.hMul (Equiv.swap x (f x)) f) y) y
                      i : Int
                      hi : Eq ((HPow.hPow f i) x) y
                      ⊢ Eq ((HMul.hMul (HPow.hPow f (HSub.hSub i 1)) (HPow.hPow f 1)) x) y
                    -/
        _ = y := by rwa [← zpow_add, sub_add_cancel]
                    /-
                      🎉 no goals
                    -/

    isCycle_swap_mul_aux₂ (i - 1) hy hi⟩


theorem IsCycle.sign {f : Perm α} (hf : IsCycle f) : sign f = -(-1) ^ #f.support :=
  let ⟨x, hx⟩ := hf
  calc
    Perm.sign f = Perm.sign (swap x (f x) * (swap x (f x) * f)) := by
      /-
        α : Type u_2
        inst✝¹ : DecidableEq α
        inst✝ : Fintype α
        f : Equiv.Perm α
        hf : f.IsCycle
        x : α
        hx : And (Ne (f x) x) (∀ ⦃y : α⦄, Ne (f y) y → f.SameCycle x y)
        ⊢ Eq (Equiv.Perm.sign f) (Equiv.Perm.sign (HMul.hMul (Equiv.swap x (f x)) (HMu …
      -/
      {rw [← mul_assoc, mul_def, mul_def, swap_swap, trans_refl]}
      /-
        🎉 no goals
      -/
    _ = -(-1) ^ #f.support :=
      if h1 : f (f x) = x then by
        have h : swap x (f x) * f = 1 := by
          simp only [mul_def, one_def]
          rw [hf.eq_swap_of_apply_apply_eq_self hx.1 h1, swap_apply_left, swap_swap]
        rw [sign_mul, sign_swap hx.1.symm, h, sign_one,
          hf.eq_swap_of_apply_apply_eq_self hx.1 h1, card_support_swap hx.1.symm]
        /-
          α : Type u_2
          inst✝¹ : DecidableEq α
          inst✝ : Fintype α
          f : Equiv.Perm α
          hf : f.IsCycle
          x : α
          hx : And (Ne (f x) x) (∀ ⦃y : α⦄, Ne (f y) y → f.SameCycle x y)
          h1 : Eq (f (f x)) x
          h : Eq (HMul.hMul (Equiv.swap x (f x)) f) 1
          ⊢ Eq (HMul.hMul (-1) 1) (Neg.neg (HPow.hPow (-1) 2))
        -/
        rfl
        /-
          🎉 no goals
        -/
      else by
        have h : #(swap x (f x) * f).support + 1 = #f.support := by
          rw [← insert_erase (mem_support.2 hx.1), support_swap_mul_eq _ _ h1,
            card_insert_of_not_mem (not_mem_erase _ _), sdiff_singleton_eq_erase]
        /-
          α : Type u_2
          inst✝¹ : DecidableEq α
          inst✝ : Fintype α
          f : Equiv.Perm α
          hf : f.IsCycle
          x : α
          hx : And (Ne (f x) x) (∀ ⦃y : α⦄, Ne (f y) y → f.SameCycle x y)
          h1 : Not (Eq (f (f x)) x)
          h : Eq (HAdd.hAdd (HMul.hMul (Equiv.swap x (f x)) f).support.card 1) f.support …
          ⊢ Eq (Equiv.Perm.sign (HMul.hMul (Equiv.swap x (f x)) (HMul.hMul (Equiv.swap x …
        -/
        have : #(swap x (f x) * f).support < #f.support := card_support_swap_mul hx.1
        /-
          α : Type u_2
          inst✝¹ : DecidableEq α
          inst✝ : Fintype α
          f : Equiv.Perm α
          hf : f.IsCycle
          x : α
          hx : And (Ne (f x) x) (∀ ⦃y : α⦄, Ne (f y) y → f.SameCycle x y)
          h1 : Not (Eq (f (f x)) x)
          h : Eq (HAdd.hAdd (HMul.hMul (Equiv.swap x (f x)) f).support.card 1) f.support …
          this : LT.lt (HMul.hMul (Equiv.swap x (f x)) f).support.card f.support.card
          ⊢ Eq (Equiv.Perm.sign (HMul.hMul (Equiv.swap x (f x)) (HMul.hMul (Equiv.swap x …
        -/
        rw [sign_mul, sign_swap hx.1.symm, (hf.swap_mul hx.1 h1).sign, ← h]
        /-
          α : Type u_2
          inst✝¹ : DecidableEq α
          inst✝ : Fintype α
          f : Equiv.Perm α
          hf : f.IsCycle
          x : α
          hx : And (Ne (f x) x) (∀ ⦃y : α⦄, Ne (f y) y → f.SameCycle x y)
          h1 : Not (Eq (f (f x)) x)
          h : Eq (HAdd.hAdd (HMul.hMul (Equiv.swap x (f x)) f).support.card 1) f.support …
          this : LT.lt (HMul.hMul (Equiv.swap x (f x)) f).support.card f.support.card
          ⊢ Eq (HMul.hMul (-1) (Neg.neg (HPow.hPow (-1) (HMul.hMul (Equiv.swap x (f x))  …
        -/
        simp only [mul_neg, neg_mul, one_mul, neg_neg, pow_add, pow_one, mul_one]
        /-
          🎉 no goals
        -/
termination_by #f.support


theorem IsCycle.of_pow {n : ℕ} (h1 : IsCycle (f ^ n)) (h2 : f.support ⊆ (f ^ n).support) :
    IsCycle f := by
  have key : ∀ x : α, (f ^ n) x ≠ x ↔ f x ≠ x := by
    simp_rw [← mem_support, ← Finset.ext_iff]
    exact (support_pow_le _ n).antisymm h2
  /-
    α : Type u_2
    f : Equiv.Perm α
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    n : Nat
    h1 : (HPow.hPow f n).IsCycle
    h2 : HasSubset.Subset f.support (HPow.hPow f n).support
    key : ∀ (x : α), Iff (Ne ((HPow.hPow f n) x) x) (Ne (f x) x)
    ⊢ f.IsCycle
  -/
  obtain ⟨x, hx1, hx2⟩ := h1
  /-
    case intro.intro
    α : Type u_2
    f : Equiv.Perm α
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    n : Nat
    h2 : HasSubset.Subset f.support (HPow.hPow f n).support
    key : ∀ (x : α), Iff (Ne ((HPow.hPow f n) x) x) (Ne (f x) x)
    x : α
    hx1 : Ne ((HPow.hPow f n) x) x
    hx2 : ∀ ⦃y : α⦄, Ne ((HPow.hPow f n) y) y → (HPow.hPow f n).SameCycle x y
    ⊢ f.IsCycle
  -/
  refine ⟨x, (key x).mp hx1, fun y hy => ?_⟩
  /-
    case intro.intro
    α : Type u_2
    f : Equiv.Perm α
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    n : Nat
    h2 : HasSubset.Subset f.support (HPow.hPow f n).support
    key : ∀ (x : α), Iff (Ne ((HPow.hPow f n) x) x) (Ne (f x) x)
    x : α
    hx1 : Ne ((HPow.hPow f n) x) x
    hx2 : ∀ ⦃y : α⦄, Ne ((HPow.hPow f n) y) y → (HPow.hPow f n).SameCycle x y
    y : α
    hy : Ne (f y) y
    ⊢ f.SameCycle x y
  -/
  cases' hx2 ((key y).mpr hy) with i _
  /-
    case intro.intro.intro
    α : Type u_2
    f : Equiv.Perm α
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    n : Nat
    h2 : HasSubset.Subset f.support (HPow.hPow f n).support
    key : ∀ (x : α), Iff (Ne ((HPow.hPow f n) x) x) (Ne (f x) x)
    x : α
    hx1 : Ne ((HPow.hPow f n) x) x
    hx2 : ∀ ⦃y : α⦄, Ne ((HPow.hPow f n) y) y → (HPow.hPow f n).SameCycle x y
    y : α
    hy : Ne (f y) y
    i : Int
    h✝ : Eq ((HPow.hPow (HPow.hPow f n) i) x) y
    ⊢ f.SameCycle x y
  -/
  exact ⟨n * i, by rwa [zpow_mul]⟩
  /-
    🎉 no goals
  -/

-- The lemma `support_zpow_le` is relevant. It means that `h2` is equivalent to
-- `σ.support = (σ ^ n).support`, as well as to `#σ.support ≤ #(σ ^ n).support`.

theorem IsCycle.of_zpow {n : ℤ} (h1 : IsCycle (f ^ n)) (h2 : f.support ⊆ (f ^ n).support) :
    IsCycle f := by
  /-
    α : Type u_2
    f : Equiv.Perm α
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    n : Int
    h1 : (HPow.hPow f n).IsCycle
    h2 : HasSubset.Subset f.support (HPow.hPow f n).support
    ⊢ f.IsCycle
  -/
  cases n
    /-
      case ofNat
      α : Type u_2
      f : Equiv.Perm α
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      a✝ : Nat
      h1 : (HPow.hPow f (Int.ofNat a✝)).IsCycle
      h2 : HasSubset.Subset f.support (HPow.hPow f (Int.ofNat a✝)).support
      ⊢ f.IsCycle
    -/
  · exact h1.of_pow h2
    /-
      🎉 no goals
    -/
    /-
      case negSucc
      α : Type u_2
      f : Equiv.Perm α
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      a✝ : Nat
      h1 : (HPow.hPow f (Int.negSucc a✝)).IsCycle
      h2 : HasSubset.Subset f.support (HPow.hPow f (Int.negSucc a✝)).support
      ⊢ f.IsCycle
    -/
  · simp only [le_eq_subset, zpow_negSucc, Perm.support_inv] at h1 h2
    /-
      case negSucc
      α : Type u_2
      f : Equiv.Perm α
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      a✝ : Nat
      h1 : (Inv.inv (HPow.hPow f (HAdd.hAdd a✝ 1))).IsCycle
      h2 : HasSubset.Subset f.support (HPow.hPow f (HAdd.hAdd a✝ 1)).support
      ⊢ f.IsCycle
    -/
    exact (inv_inv (f ^ _) ▸ h1.inv).of_pow h2
    /-
      🎉 no goals
    -/


theorem nodup_of_pairwise_disjoint_cycles {l : List (Perm β)} (h1 : ∀ f ∈ l, IsCycle f)
    (h2 : l.Pairwise Disjoint) : l.Nodup :=
  nodup_of_pairwise_disjoint (fun h => (h1 1 h).ne_one rfl) h2


/-- Unlike `support_congr`, which assumes that `∀ (x ∈ g.support), f x = g x)`, here
we have the weaker assumption that `∀ (x ∈ f.support), f x = g x`. -/
theorem IsCycle.support_congr (hf : IsCycle f) (hg : IsCycle g) (h : f.support ⊆ g.support)
    (h' : ∀ x ∈ f.support, f x = g x) : f = g := by
  have : f.support = g.support := by
    refine le_antisymm h ?_
    intro z hz
    obtain ⟨x, hx, _⟩ := id hf
    have hx' : g x ≠ x := by rwa [← h' x (mem_support.mpr hx)]
    obtain ⟨m, hm⟩ := hg.exists_pow_eq hx' (mem_support.mp hz)
    have h'' : ∀ x ∈ f.support ∩ g.support, f x = g x := by
      intro x hx
      exact h' x (mem_of_mem_inter_left hx)
    rwa [← hm, ←
      pow_eq_on_of_mem_support h'' _ x
        (mem_inter_of_mem (mem_support.mpr hx) (mem_support.mpr hx')),
      pow_apply_mem_support, mem_support]
  /-
    α : Type u_2
    f g : Equiv.Perm α
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    hf : f.IsCycle
    hg : g.IsCycle
    h : HasSubset.Subset f.support g.support
    h' : ∀ (x : α), Membership.mem f.support x → Eq (f x) (g x)
    this : Eq f.support g.support
    ⊢ Eq f g
  -/
  refine Equiv.Perm.support_congr h ?_
  /-
    α : Type u_2
    f g : Equiv.Perm α
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    hf : f.IsCycle
    hg : g.IsCycle
    h : HasSubset.Subset f.support g.support
    h' : ∀ (x : α), Membership.mem f.support x → Eq (f x) (g x)
    this : Eq f.support g.support
    ⊢ ∀ (x : α), Membership.mem g.support x → Eq (f x) (g x)
  -/
  simpa [← this] using h'
  /-
    🎉 no goals
  -/


/-- If two cyclic permutations agree on all terms in their intersection,
and that intersection is not empty, then the two cyclic permutations must be equal. -/
theorem IsCycle.eq_on_support_inter_nonempty_congr (hf : IsCycle f) (hg : IsCycle g)
    (h : ∀ x ∈ f.support ∩ g.support, f x = g x)
    (hx : f x = g x) (hx' : x ∈ f.support) : f = g := by
  /-
    α : Type u_2
    f g : Equiv.Perm α
    x : α
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    hf : f.IsCycle
    hg : g.IsCycle
    h : ∀ (x : α), Membership.mem (Inter.inter f.support g.support) x → Eq (f x) ( …
    hx : Eq (f x) (g x)
    hx' : Membership.mem f.support x
    ⊢ Eq f g
  -/
  have hx'' : x ∈ g.support := by rwa [mem_support, ← hx, ← mem_support]
  have : f.support ⊆ g.support := by
    intro y hy
    obtain ⟨k, rfl⟩ := hf.exists_pow_eq (mem_support.mp hx') (mem_support.mp hy)
    rwa [pow_eq_on_of_mem_support h _ _ (mem_inter_of_mem hx' hx''), pow_apply_mem_support]
  /-
    α : Type u_2
    f g : Equiv.Perm α
    x : α
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    hf : f.IsCycle
    hg : g.IsCycle
    h : ∀ (x : α), Membership.mem (Inter.inter f.support g.support) x → Eq (f x) ( …
    hx : Eq (f x) (g x)
    hx' : Membership.mem f.support x
    hx'' : Membership.mem g.support x
    this : HasSubset.Subset f.support g.support
    ⊢ Eq f g
  -/
  rw [inter_eq_left.mpr this] at h
  /-
    α : Type u_2
    f g : Equiv.Perm α
    x : α
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    hf : f.IsCycle
    hg : g.IsCycle
    h : ∀ (x : α), Membership.mem f.support x → Eq (f x) (g x)
    hx : Eq (f x) (g x)
    hx' : Membership.mem f.support x
    hx'' : Membership.mem g.support x
    this : HasSubset.Subset f.support g.support
    ⊢ Eq f g
  -/
  exact hf.support_congr hg this h
  /-
    🎉 no goals
  -/


theorem IsCycle.support_pow_eq_iff (hf : IsCycle f) {n : ℕ} :
    support (f ^ n) = support f ↔ ¬orderOf f ∣ n := by
  /-
    α : Type u_2
    f : Equiv.Perm α
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    hf : f.IsCycle
    n : Nat
    ⊢ Iff (Eq (HPow.hPow f n).support f.support) (Not (Dvd.dvd (orderOf f) n))
  -/
  rw [orderOf_dvd_iff_pow_eq_one]
  /-
    α : Type u_2
    f : Equiv.Perm α
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    hf : f.IsCycle
    n : Nat
    ⊢ Iff (Eq (HPow.hPow f n).support f.support) (Not (Eq (HPow.hPow f n) 1))
  -/
  constructor
    /-
      case mp
      α : Type u_2
      f : Equiv.Perm α
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      hf : f.IsCycle
      n : Nat
      ⊢ Eq (HPow.hPow f n).support f.support → Not (Eq (HPow.hPow f n) 1)
    -/
  · intro h H
    /-
      case mp
      α : Type u_2
      f : Equiv.Perm α
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      hf : f.IsCycle
      n : Nat
      h : Eq (HPow.hPow f n).support f.support
      H : Eq (HPow.hPow f n) 1
      ⊢ False
    -/
    refine hf.ne_one ?_
    /-
      case mp
      α : Type u_2
      f : Equiv.Perm α
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      hf : f.IsCycle
      n : Nat
      h : Eq (HPow.hPow f n).support f.support
      H : Eq (HPow.hPow f n) 1
      ⊢ Eq f 1
    -/
    rw [← support_eq_empty_iff, ← h, H, support_one]
    /-
      🎉 no goals
    -/
    /-
      case mpr
      α : Type u_2
      f : Equiv.Perm α
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      hf : f.IsCycle
      n : Nat
      ⊢ Not (Eq (HPow.hPow f n) 1) → Eq (HPow.hPow f n).support f.support
    -/
  · intro H
    /-
      case mpr
      α : Type u_2
      f : Equiv.Perm α
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      hf : f.IsCycle
      n : Nat
      H : Not (Eq (HPow.hPow f n) 1)
      ⊢ Eq (HPow.hPow f n).support f.support
    -/
    apply le_antisymm (support_pow_le _ n) _
    /-
      α : Type u_2
      f : Equiv.Perm α
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      hf : f.IsCycle
      n : Nat
      H : Not (Eq (HPow.hPow f n) 1)
      ⊢ LE.le f.support (HPow.hPow f n).support
    -/
    intro x hx
    /-
      α : Type u_2
      f : Equiv.Perm α
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      hf : f.IsCycle
      n : Nat
      H : Not (Eq (HPow.hPow f n) 1)
      x : α
      hx : Membership.mem f.support x
      ⊢ Membership.mem (HPow.hPow f n).support x
    -/
    contrapose! H
    /-
      α : Type u_2
      f : Equiv.Perm α
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      hf : f.IsCycle
      n : Nat
      x : α
      hx : Membership.mem f.support x
      H : Not (Membership.mem (HPow.hPow f n).support x)
      ⊢ Eq (HPow.hPow f n) 1
    -/
    ext z
    /-
      case H
      α : Type u_2
      f : Equiv.Perm α
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      hf : f.IsCycle
      n : Nat
      x : α
      hx : Membership.mem f.support x
      H : Not (Membership.mem (HPow.hPow f n).support x)
      z : α
      ⊢ Eq ((HPow.hPow f n) z) (1 z)
    -/
    by_cases hz : f z = z
      /-
        case pos
        α : Type u_2
        f : Equiv.Perm α
        inst✝¹ : DecidableEq α
        inst✝ : Fintype α
        hf : f.IsCycle
        n : Nat
        x : α
        hx : Membership.mem f.support x
        H : Not (Membership.mem (HPow.hPow f n).support x)
        z : α
        hz : Eq (f z) z
        ⊢ Eq ((HPow.hPow f n) z) (1 z)
      -/
    · rw [pow_apply_eq_self_of_apply_eq_self hz, one_apply]
      /-
        🎉 no goals
      -/
      /-
        case neg
        α : Type u_2
        f : Equiv.Perm α
        inst✝¹ : DecidableEq α
        inst✝ : Fintype α
        hf : f.IsCycle
        n : Nat
        x : α
        hx : Membership.mem f.support x
        H : Not (Membership.mem (HPow.hPow f n).support x)
        z : α
        hz : Not (Eq (f z) z)
        ⊢ Eq ((HPow.hPow f n) z) (1 z)
      -/
    · obtain ⟨k, rfl⟩ := hf.exists_pow_eq hz (mem_support.mp hx)
      /-
        case neg.intro
        α : Type u_2
        f : Equiv.Perm α
        inst✝¹ : DecidableEq α
        inst✝ : Fintype α
        hf : f.IsCycle
        n : Nat
        z : α
        hz : Not (Eq (f z) z)
        k : Nat
        hx : Membership.mem f.support ((HPow.hPow f k) z)
        H : Not (Membership.mem (HPow.hPow f n).support ((HPow.hPow f k) z))
        ⊢ Eq ((HPow.hPow f n) z) (1 z)
      -/
      apply (f ^ k).injective
      /-
        case neg.intro.a
        α : Type u_2
        f : Equiv.Perm α
        inst✝¹ : DecidableEq α
        inst✝ : Fintype α
        hf : f.IsCycle
        n : Nat
        z : α
        hz : Not (Eq (f z) z)
        k : Nat
        hx : Membership.mem f.support ((HPow.hPow f k) z)
        H : Not (Membership.mem (HPow.hPow f n).support ((HPow.hPow f k) z))
        ⊢ Eq ((HPow.hPow f k) ((HPow.hPow f n) z)) ((HPow.hPow f k) (1 z))
      -/
      rw [← mul_apply, (Commute.pow_pow_self _ _ _).eq, mul_apply]
      /-
        case neg.intro.a
        α : Type u_2
        f : Equiv.Perm α
        inst✝¹ : DecidableEq α
        inst✝ : Fintype α
        hf : f.IsCycle
        n : Nat
        z : α
        hz : Not (Eq (f z) z)
        k : Nat
        hx : Membership.mem f.support ((HPow.hPow f k) z)
        H : Not (Membership.mem (HPow.hPow f n).support ((HPow.hPow f k) z))
        ⊢ Eq ((HPow.hPow f n) ((HPow.hPow f k) z)) ((HPow.hPow f k) (1 z))
      -/
      simpa using H
      /-
        🎉 no goals
      -/


theorem IsCycle.support_pow_of_pos_of_lt_orderOf (hf : IsCycle f) {n : ℕ} (npos : 0 < n)
    (hn : n < orderOf f) : (f ^ n).support = f.support :=
  hf.support_pow_eq_iff.2 <| Nat.not_dvd_of_pos_of_lt npos hn


theorem IsCycle.pow_iff [Finite β] {f : Perm β} (hf : IsCycle f) {n : ℕ} :
    IsCycle (f ^ n) ↔ n.Coprime (orderOf f) := by
  classical
    cases nonempty_fintype β
    constructor
    · intro h
      have hr : support (f ^ n) = support f := by
        rw [hf.support_pow_eq_iff]
        rintro ⟨k, rfl⟩
        refine h.ne_one ?_
        simp [pow_mul, pow_orderOf_eq_one]
      have : orderOf (f ^ n) = orderOf f := by rw [h.orderOf, hr, hf.orderOf]
      rw [orderOf_pow, Nat.div_eq_self] at this
      cases' this with h
      · exact absurd h (orderOf_pos _).ne'
      · rwa [Nat.coprime_iff_gcd_eq_one, Nat.gcd_comm]
    · intro h
      obtain ⟨m, hm⟩ := exists_pow_eq_self_of_coprime h
      have hf' : IsCycle ((f ^ n) ^ m) := by rwa [hm]
      refine hf'.of_pow fun x hx => ?_
      rw [hm]
      exact support_pow_le _ n hx

-- TODO: Define a `Set`-valued support to get rid of the `Finite β` assumption

theorem IsCycle.pow_eq_one_iff [Finite β] {f : Perm β} (hf : IsCycle f) {n : ℕ} :
    f ^ n = 1 ↔ ∃ x, f x ≠ x ∧ (f ^ n) x = x := by
  classical
    cases nonempty_fintype β
    constructor
    · intro h
      obtain ⟨x, hx, -⟩ := id hf
      exact ⟨x, hx, by simp [h]⟩
    · rintro ⟨x, hx, hx'⟩
      by_cases h : support (f ^ n) = support f
      · rw [← mem_support, ← h, mem_support] at hx
        contradiction
      · rw [hf.support_pow_eq_iff, Classical.not_not] at h
        obtain ⟨k, rfl⟩ := h
        rw [pow_mul, pow_orderOf_eq_one, one_pow]

-- TODO: Define a `Set`-valued support to get rid of the `Finite β` assumption

theorem IsCycle.pow_eq_one_iff' [Finite β] {f : Perm β} (hf : IsCycle f) {n : ℕ} {x : β}
    (hx : f x ≠ x) : f ^ n = 1 ↔ (f ^ n) x = x :=
  ⟨fun h => DFunLike.congr_fun h x, fun h => hf.pow_eq_one_iff.2 ⟨x, hx, h⟩⟩

-- TODO: Define a `Set`-valued support to get rid of the `Finite β` assumption

theorem IsCycle.pow_eq_one_iff'' [Finite β] {f : Perm β} (hf : IsCycle f) {n : ℕ} :
    f ^ n = 1 ↔ ∀ x, f x ≠ x → (f ^ n) x = x :=
  ⟨fun h _ hx => (hf.pow_eq_one_iff' hx).1 h, fun h =>
    let ⟨_, hx, _⟩ := id hf
    (hf.pow_eq_one_iff' hx).2 (h _ hx)⟩

-- TODO: Define a `Set`-valued support to get rid of the `Finite β` assumption

theorem IsCycle.pow_eq_pow_iff [Finite β] {f : Perm β} (hf : IsCycle f) {a b : ℕ} :
    f ^ a = f ^ b ↔ ∃ x, f x ≠ x ∧ (f ^ a) x = (f ^ b) x := by
  classical
    cases nonempty_fintype β
    constructor
    · intro h
      obtain ⟨x, hx, -⟩ := id hf
      exact ⟨x, hx, by simp [h]⟩
    · rintro ⟨x, hx, hx'⟩
      wlog hab : a ≤ b generalizing a b
      · exact (this hx'.symm (le_of_not_le hab)).symm
      suffices f ^ (b - a) = 1 by
        rw [pow_sub _ hab, mul_inv_eq_one] at this
        rw [this]
      rw [hf.pow_eq_one_iff]
      by_cases hfa : (f ^ a) x ∈ f.support
      · refine ⟨(f ^ a) x, mem_support.mp hfa, ?_⟩
        simp only [pow_sub _ hab, Equiv.Perm.coe_mul, Function.comp_apply, inv_apply_self, ← hx']
      · have h := @Equiv.Perm.zpow_apply_comm _ f 1 a x
        simp only [zpow_one, zpow_natCast] at h
        rw [not_mem_support, h, Function.Injective.eq_iff (f ^ a).injective] at hfa
        contradiction


theorem IsCycle.isCycle_pow_pos_of_lt_prime_order [Finite β] {f : Perm β} (hf : IsCycle f)
    (hf' : (orderOf f).Prime) (n : ℕ) (hn : 0 < n) (hn' : n < orderOf f) : IsCycle (f ^ n) := by
  classical
    cases nonempty_fintype β
    have : n.Coprime (orderOf f) := by
      refine Nat.Coprime.symm ?_
      rw [Nat.Prime.coprime_iff_not_dvd hf']
      exact Nat.not_dvd_of_pos_of_lt hn hn'
    obtain ⟨m, hm⟩ := exists_pow_eq_self_of_coprime this
    have hf'' := hf
    rw [← hm] at hf''
    refine hf''.of_pow ?_
    rw [hm]
    exact support_pow_le f n


theorem _root_.Int.addLeft_one_isCycle : (Equiv.addLeft 1 : Perm ℤ).IsCycle :=
                                     /-
                                       n : Int
                                       x✝ : Ne ((Equiv.addLeft 1) n) n
                                       ⊢ Eq ((HPow.hPow (Equiv.addLeft 1) n) 0) n
                                     -/
  ⟨0, one_ne_zero, fun n _ => ⟨n, by simp⟩⟩
                                     /-
                                       🎉 no goals
                                     -/


theorem _root_.Int.addRight_one_isCycle : (Equiv.addRight 1 : Perm ℤ).IsCycle :=
                                     /-
                                       n : Int
                                       x✝ : Ne ((Equiv.addRight 1) n) n
                                       ⊢ Eq ((HPow.hPow (Equiv.addRight 1) n) 0) n
                                     -/
  ⟨0, one_ne_zero, fun n _ => ⟨n, by simp⟩⟩
                                     /-
                                       🎉 no goals
                                     -/


theorem IsCycle.isConj (hσ : IsCycle σ) (hτ : IsCycle τ) (h : #σ.support = #τ.support) :
    IsConj σ τ := by
  refine
    isConj_of_support_equiv
      (hσ.zpowersEquivSupport.symm.trans <|
        (zpowersEquivZPowers <| by rw [hσ.orderOf, h, hτ.orderOf]).trans hτ.zpowersEquivSupport)
      ?_
  /-
    α : Type u_2
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    σ τ : Equiv.Perm α
    hσ : σ.IsCycle
    hτ : τ.IsCycle
    h : Eq σ.support.card τ.support.card
    ⊢ ∀ (x : α) (hx : Membership.mem (↑σ.support) x), Eq (↑((hσ.zpowersEquivSuppor …
  -/
  intro x hx
  /-
    α : Type u_2
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    σ τ : Equiv.Perm α
    hσ : σ.IsCycle
    hτ : τ.IsCycle
    h : Eq σ.support.card τ.support.card
    x : α
    hx : Membership.mem (↑σ.support) x
    ⊢ Eq (↑((hσ.zpowersEquivSupport.symm.trans ((zpowersEquivZPowers ⋯).trans hτ.z …
  -/
  simp only [Perm.mul_apply, Equiv.trans_apply, Equiv.sumCongr_apply]
  /-
    α : Type u_2
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    σ τ : Equiv.Perm α
    hσ : σ.IsCycle
    hτ : τ.IsCycle
    h : Eq σ.support.card τ.support.card
    x : α
    hx : Membership.mem (↑σ.support) x
    ⊢ Eq (↑(hτ.zpowersEquivSupport ((zpowersEquivZPowers ⋯) (hσ.zpowersEquivSuppor …
  -/
  obtain ⟨n, rfl⟩ := hσ.exists_pow_eq (Classical.choose_spec hσ).1 (mem_support.1 hx)
  /-
    case intro
    α : Type u_2
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    σ τ : Equiv.Perm α
    hσ : σ.IsCycle
    hτ : τ.IsCycle
    h : Eq σ.support.card τ.support.card
    n : Nat
    hx : Membership.mem (↑σ.support) ((HPow.hPow σ n) (Classical.choose hσ))
    ⊢ Eq (↑(hτ.zpowersEquivSupport ((zpowersEquivZPowers ⋯) (hσ.zpowersEquivSuppor …
  -/
  erw [hσ.zpowersEquivSupport_symm_apply n]
  /-
    case intro
    α : Type u_2
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    σ τ : Equiv.Perm α
    hσ : σ.IsCycle
    hτ : τ.IsCycle
    h : Eq σ.support.card τ.support.card
    n : Nat
    hx : Membership.mem (↑σ.support) ((HPow.hPow σ n) (Classical.choose hσ))
    ⊢ Eq (↑(hτ.zpowersEquivSupport ((zpowersEquivZPowers ⋯) (hσ.zpowersEquivSuppor …
  -/
  simp only [← Perm.mul_apply, ← pow_succ']
  /-
    case intro
    α : Type u_2
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    σ τ : Equiv.Perm α
    hσ : σ.IsCycle
    hτ : τ.IsCycle
    h : Eq σ.support.card τ.support.card
    n : Nat
    hx : Membership.mem (↑σ.support) ((HPow.hPow σ n) (Classical.choose hσ))
    ⊢ Eq (↑(hτ.zpowersEquivSupport ((zpowersEquivZPowers ⋯) (hσ.zpowersEquivSuppor …
  -/
  erw [hσ.zpowersEquivSupport_symm_apply (n + 1)]
  -- This used to be a `simp only` before https://github.com/leanprover/lean4/pull/2644
  /-
    case intro
    α : Type u_2
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    σ τ : Equiv.Perm α
    hσ : σ.IsCycle
    hτ : τ.IsCycle
    h : Eq σ.support.card τ.support.card
    n : Nat
    hx : Membership.mem (↑σ.support) ((HPow.hPow σ n) (Classical.choose hσ))
    ⊢ Eq (↑(hτ.zpowersEquivSupport ((zpowersEquivZPowers ⋯) ⟨HPow.hPow σ (HAdd.hAd …
  -/
  erw [zpowersEquivZPowers_apply, zpowersEquivZPowers_apply, zpowersEquivSupport_apply]
  -- This used to be `rw`, but we need `erw` after https://github.com/leanprover/lean4/pull/2644
  /-
    case intro
    α : Type u_2
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    σ τ : Equiv.Perm α
    hσ : σ.IsCycle
    hτ : τ.IsCycle
    h : Eq σ.support.card τ.support.card
    n : Nat
    hx : Membership.mem (↑σ.support) ((HPow.hPow σ n) (Classical.choose hσ))
    ⊢ Eq (↑⟨(HPow.hPow τ (HAdd.hAdd n 1)) (Classical.choose hτ), ⋯⟩) (τ ↑(hτ.zpowe …
  -/
  simp_rw [pow_succ', Perm.mul_apply]
  /-
    case intro
    α : Type u_2
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    σ τ : Equiv.Perm α
    hσ : σ.IsCycle
    hτ : τ.IsCycle
    h : Eq σ.support.card τ.support.card
    n : Nat
    hx : Membership.mem (↑σ.support) ((HPow.hPow σ n) (Classical.choose hσ))
    ⊢ Eq (τ ((HPow.hPow τ n) (Classical.choose hτ))) (τ ↑(hτ.zpowersEquivSupport ⟨ …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem IsCycle.isConj_iff (hσ : IsCycle σ) (hτ : IsCycle τ) :
    IsConj σ τ ↔ #σ.support = #τ.support where
  mp h := by
    /-
      α : Type u_2
      inst✝¹ : Fintype α
      inst✝ : DecidableEq α
      σ τ : Equiv.Perm α
      hσ : σ.IsCycle
      hτ : τ.IsCycle
      h : IsConj σ τ
      ⊢ Eq σ.support.card τ.support.card
    -/
    obtain ⟨π, rfl⟩ := (_root_.isConj_iff).1 h
    refine Finset.card_bij (fun a _ => π a) (fun _ ha => ?_) (fun _ _ _ _ ab => π.injective ab)
        fun b hb ↦ ⟨π⁻¹ b, ?_, π.apply_inv_self b⟩
      /-
        case intro.refine_1
        α : Type u_2
        inst✝¹ : Fintype α
        inst✝ : DecidableEq α
        σ : Equiv.Perm α
        hσ : σ.IsCycle
        π : Equiv.Perm α
        hτ : (HMul.hMul (HMul.hMul π σ) (Inv.inv π)).IsCycle
        h : IsConj σ (HMul.hMul (HMul.hMul π σ) (Inv.inv π))
        x✝ : α
        ha : Membership.mem σ.support x✝
        ⊢ Membership.mem (HMul.hMul (HMul.hMul π σ) (Inv.inv π)).support ((fun a x =>  …
      -/
    · simp [mem_support.1 ha]
      /-
        🎉 no goals
      -/
    /-
      case intro.refine_2
      α : Type u_2
      inst✝¹ : Fintype α
      inst✝ : DecidableEq α
      σ : Equiv.Perm α
      hσ : σ.IsCycle
      π : Equiv.Perm α
      hτ : (HMul.hMul (HMul.hMul π σ) (Inv.inv π)).IsCycle
      h : IsConj σ (HMul.hMul (HMul.hMul π σ) (Inv.inv π))
      b : α
      hb : Membership.mem (HMul.hMul (HMul.hMul π σ) (Inv.inv π)).support b
      ⊢ Membership.mem σ.support ((Inv.inv π) b)
    -/
    contrapose! hb
    /-
      case intro.refine_2
      α : Type u_2
      inst✝¹ : Fintype α
      inst✝ : DecidableEq α
      σ : Equiv.Perm α
      hσ : σ.IsCycle
      π : Equiv.Perm α
      hτ : (HMul.hMul (HMul.hMul π σ) (Inv.inv π)).IsCycle
      h : IsConj σ (HMul.hMul (HMul.hMul π σ) (Inv.inv π))
      b : α
      hb : Not (Membership.mem σ.support ((Inv.inv π) b))
      ⊢ Not (Membership.mem (HMul.hMul (HMul.hMul π σ) (Inv.inv π)).support b)
    -/
    rw [mem_support, Classical.not_not] at hb
    /-
      case intro.refine_2
      α : Type u_2
      inst✝¹ : Fintype α
      inst✝ : DecidableEq α
      σ : Equiv.Perm α
      hσ : σ.IsCycle
      π : Equiv.Perm α
      hτ : (HMul.hMul (HMul.hMul π σ) (Inv.inv π)).IsCycle
      h : IsConj σ (HMul.hMul (HMul.hMul π σ) (Inv.inv π))
      b : α
      hb : Eq (σ ((Inv.inv π) b)) ((Inv.inv π) b)
      ⊢ Not (Membership.mem (HMul.hMul (HMul.hMul π σ) (Inv.inv π)).support b)
    -/
    rw [mem_support, Classical.not_not, Perm.mul_apply, Perm.mul_apply, hb, Perm.apply_inv_self]
    /-
      🎉 no goals
    -/
  mpr := hσ.isConj hτ


/-- A permutation is a cycle on `s` when any two points of `s` are related by repeated application
of the permutation. Note that this means the identity is a cycle of subsingleton sets. -/
def IsCycleOn (f : Perm α) (s : Set α) : Prop :=
  Set.BijOn f s s ∧ ∀ ⦃x⦄, x ∈ s → ∀ ⦃y⦄, y ∈ s → f.SameCycle x y


@[simp]
                                              /-
                                                α : Type u_2
                                                f : Equiv.Perm α
                                                ⊢ f.IsCycleOn EmptyCollection.emptyCollection
                                              -/
theorem isCycleOn_empty : f.IsCycleOn ∅ := by simp [IsCycleOn, Set.bijOn_empty]
                                              /-
                                                🎉 no goals
                                              -/


@[simp]
theorem isCycleOn_one : (1 : Perm α).IsCycleOn s ↔ s.Subsingleton := by
  /-
    α : Type u_2
    s : Set α
    ⊢ Iff (Equiv.Perm.IsCycleOn 1 s) s.Subsingleton
  -/
  simp [IsCycleOn, Set.bijOn_id, Set.Subsingleton]
  /-
    🎉 no goals
  -/


alias ⟨IsCycleOn.subsingleton, _root_.Set.Subsingleton.isCycleOn_one⟩ := isCycleOn_one


@[simp]
                                                              /-
                                                                α : Type u_2
                                                                f : Equiv.Perm α
                                                                a : α
                                                                ⊢ Iff (f.IsCycleOn (Singleton.singleton a)) (Eq (f a) a)
                                                              -/
theorem isCycleOn_singleton : f.IsCycleOn {a} ↔ f a = a := by simp [IsCycleOn, SameCycle.rfl]
                                                              /-
                                                                🎉 no goals
                                                              -/


theorem isCycleOn_of_subsingleton [Subsingleton α] (f : Perm α) (s : Set α) : f.IsCycleOn s :=
  ⟨s.bijOn_of_subsingleton _, fun x _ y _ => (Subsingleton.elim x y).sameCycle _⟩


@[simp]
theorem isCycleOn_inv : f⁻¹.IsCycleOn s ↔ f.IsCycleOn s := by
  /-
    α : Type u_2
    f : Equiv.Perm α
    s : Set α
    ⊢ Iff ((Inv.inv f).IsCycleOn s) (f.IsCycleOn s)
  -/
  simp only [IsCycleOn, sameCycle_inv, and_congr_left_iff]
  /-
    α : Type u_2
    f : Equiv.Perm α
    s : Set α
    ⊢ (∀ ⦃x : α⦄, Membership.mem s x → ∀ ⦃y : α⦄, Membership.mem s y → f.SameCycle …
  -/
  exact fun _ ↦ ⟨fun h ↦ Set.BijOn.perm_inv h, fun h ↦ Set.BijOn.perm_inv h⟩
  /-
    🎉 no goals
  -/


alias ⟨IsCycleOn.of_inv, IsCycleOn.inv⟩ := isCycleOn_inv


theorem IsCycleOn.conj (h : f.IsCycleOn s) : (g * f * g⁻¹).IsCycleOn ((g : Perm α) '' s) :=
  ⟨(g.bijOn_image.comp h.1).comp g.bijOn_symm_image, fun x hx y hy => by
    /-
      α : Type u_2
      f g : Equiv.Perm α
      s : Set α
      h : f.IsCycleOn s
      x : α
      hx : Membership.mem (_root_.Set.image (⇑g) s) x
      y : α
      hy : Membership.mem (_root_.Set.image (⇑g) s) y
      ⊢ (HMul.hMul (HMul.hMul g f) (Inv.inv g)).SameCycle x y
    -/
    rw [← preimage_inv] at hx hy
    /-
      α : Type u_2
      f g : Equiv.Perm α
      s : Set α
      h : f.IsCycleOn s
      x : α
      hx : Membership.mem (Set.preimage (⇑(Inv.inv g)) s) x
      y : α
      hy : Membership.mem (Set.preimage (⇑(Inv.inv g)) s) y
      ⊢ (HMul.hMul (HMul.hMul g f) (Inv.inv g)).SameCycle x y
    -/
                                                               /-
                                                                 🎉 no goals
                                                               -/
    convert Equiv.Perm.SameCycle.conj (h.2 hx hy) (g := g) <;> rw [apply_inv_self]⟩
                                                               /-
                                                                 🎉 no goals
                                                               -/


theorem isCycleOn_swap [DecidableEq α] (hab : a ≠ b) : (swap a b).IsCycleOn {a, b} :=
                  /-
                    α : Type u_2
                    a b : α
                    inst✝ : DecidableEq α
                    hab : Ne a b
                    ⊢ Membership.mem (Insert.insert a (Singleton.singleton b)) a
                  -/
                  /-
                    🎉 no goals
                  -/
  ⟨bijOn_swap (by simp) (by simp), fun x hx y hy => by
                            /-
                              🎉 no goals
                            -/
    /-
      α : Type u_2
      a b : α
      inst✝ : DecidableEq α
      hab : Ne a b
      x : α
      hx : Membership.mem (Insert.insert a (Singleton.singleton b)) x
      y : α
      hy : Membership.mem (Insert.insert a (Singleton.singleton b)) y
      ⊢ (Equiv.swap a b).SameCycle x y
    -/
    rw [Set.mem_insert_iff, Set.mem_singleton_iff] at hx hy
    /-
      α : Type u_2
      a b : α
      inst✝ : DecidableEq α
      hab : Ne a b
      x : α
      hx : Or (Eq x a) (Eq x b)
      y : α
      hy : Or (Eq y a) (Eq y b)
      ⊢ (Equiv.swap a b).SameCycle x y
    -/
    obtain rfl | rfl := hx <;> obtain rfl | rfl := hy
      /-
        case inl.inl
        α : Type u_2
        b : α
        inst✝ : DecidableEq α
        y : α
        hab : Ne y b
        ⊢ (Equiv.swap y b).SameCycle y y
      -/
    · exact ⟨0, by rw [zpow_zero, coe_one, id]⟩
      /-
        🎉 no goals
      -/
      /-
        case inl.inr
        α : Type u_2
        inst✝ : DecidableEq α
        x y : α
        hab : Ne x y
        ⊢ (Equiv.swap x y).SameCycle x y
      -/
    · exact ⟨1, by rw [zpow_one, swap_apply_left]⟩
      /-
        🎉 no goals
      -/
      /-
        case inr.inl
        α : Type u_2
        inst✝ : DecidableEq α
        x y : α
        hab : Ne y x
        ⊢ (Equiv.swap y x).SameCycle x y
      -/
    · exact ⟨1, by rw [zpow_one, swap_apply_right]⟩
      /-
        🎉 no goals
      -/
      /-
        case inr.inr
        α : Type u_2
        a : α
        inst✝ : DecidableEq α
        y : α
        hab : Ne a y
        ⊢ (Equiv.swap a y).SameCycle y y
      -/
    · exact ⟨0, by rw [zpow_zero, coe_one, id]⟩⟩
      /-
        🎉 no goals
      -/


protected theorem IsCycleOn.apply_ne (hf : f.IsCycleOn s) (hs : s.Nontrivial) (ha : a ∈ s) :
    f a ≠ a := by
  /-
    α : Type u_2
    f : Equiv.Perm α
    s : Set α
    a : α
    hf : f.IsCycleOn s
    hs : s.Nontrivial
    ha : Membership.mem s a
    ⊢ Ne (f a) a
  -/
  obtain ⟨b, hb, hba⟩ := hs.exists_ne a
  /-
    case intro.intro
    α : Type u_2
    f : Equiv.Perm α
    s : Set α
    a : α
    hf : f.IsCycleOn s
    hs : s.Nontrivial
    ha : Membership.mem s a
    b : α
    hb : Membership.mem s b
    hba : Ne b a
    ⊢ Ne (f a) a
  -/
  obtain ⟨n, rfl⟩ := hf.2 ha hb
  /-
    case intro.intro.intro
    α : Type u_2
    f : Equiv.Perm α
    s : Set α
    a : α
    hf : f.IsCycleOn s
    hs : s.Nontrivial
    ha : Membership.mem s a
    n : Int
    hb : Membership.mem s ((HPow.hPow f n) a)
    hba : Ne ((HPow.hPow f n) a) a
    ⊢ Ne (f a) a
  -/
  exact fun h => hba (IsFixedPt.perm_zpow h n)
  /-
    🎉 no goals
  -/


protected theorem IsCycle.isCycleOn (hf : f.IsCycle) : f.IsCycleOn { x | f x ≠ x } :=
  ⟨f.bijOn fun _ => f.apply_eq_iff_eq.not, fun _ ha _ => hf.sameCycle ha⟩


/-- This lemma demonstrates the relation between `Equiv.Perm.IsCycle` and `Equiv.Perm.IsCycleOn`
in non-degenerate cases. -/
theorem isCycle_iff_exists_isCycleOn :
    f.IsCycle ↔ ∃ s : Set α, s.Nontrivial ∧ f.IsCycleOn s ∧ ∀ ⦃x⦄, ¬IsFixedPt f x → x ∈ s := by
  /-
    α : Type u_2
    f : Equiv.Perm α
    ⊢ Iff f.IsCycle (Exists fun s => And s.Nontrivial (And (f.IsCycleOn s) (∀ ⦃x : …
  -/
  refine ⟨fun hf => ⟨{ x | f x ≠ x }, ?_, hf.isCycleOn, fun _ => id⟩, ?_⟩
    /-
      case refine_1
      α : Type u_2
      f : Equiv.Perm α
      hf : f.IsCycle
      ⊢ (setOf fun x => Ne (f x) x).Nontrivial
    -/
  · obtain ⟨a, ha⟩ := hf
    /-
      case refine_1.intro
      α : Type u_2
      f : Equiv.Perm α
      a : α
      ha : And (Ne (f a) a) (∀ ⦃y : α⦄, Ne (f y) y → f.SameCycle a y)
      ⊢ (setOf fun x => Ne (f x) x).Nontrivial
    -/
    exact ⟨f a, f.injective.ne ha.1, a, ha.1, ha.1⟩
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_2
      f : Equiv.Perm α
      ⊢ (Exists fun s => And s.Nontrivial (And (f.IsCycleOn s) (∀ ⦃x : α⦄, Not (Func …
    -/
  · rintro ⟨s, hs, hf, hsf⟩
    /-
      case refine_2.intro.intro.intro
      α : Type u_2
      f : Equiv.Perm α
      s : Set α
      hs : s.Nontrivial
      hf : f.IsCycleOn s
      hsf : ∀ ⦃x : α⦄, Not (Function.IsFixedPt (⇑f) x) → Membership.mem s x
      ⊢ f.IsCycle
    -/
    obtain ⟨a, ha⟩ := hs.nonempty
    /-
      case refine_2.intro.intro.intro.intro
      α : Type u_2
      f : Equiv.Perm α
      s : Set α
      hs : s.Nontrivial
      hf : f.IsCycleOn s
      hsf : ∀ ⦃x : α⦄, Not (Function.IsFixedPt (⇑f) x) → Membership.mem s x
      a : α
      ha : Membership.mem s a
      ⊢ f.IsCycle
    -/
    exact ⟨a, hf.apply_ne hs ha, fun b hb => hf.2 ha <| hsf hb⟩
    /-
      🎉 no goals
    -/


theorem IsCycleOn.apply_mem_iff (hf : f.IsCycleOn s) : f x ∈ s ↔ x ∈ s :=
  ⟨fun hx => by
    /-
      α : Type u_2
      f : Equiv.Perm α
      s : Set α
      x : α
      hf : f.IsCycleOn s
      hx : Membership.mem s (f x)
      ⊢ Membership.mem s x
    -/
    convert hf.1.perm_inv.1 hx
    /-
      case h.e'_5
      α : Type u_2
      f : Equiv.Perm α
      s : Set α
      x : α
      hf : f.IsCycleOn s
      hx : Membership.mem s (f x)
      ⊢ Eq x ((Inv.inv f) (f x))
    -/
    rw [inv_apply_self], fun hx => hf.1.mapsTo hx⟩
    /-
      🎉 no goals
    -/


/-- Note that the identity satisfies `IsCycleOn` for any subsingleton set, but not `IsCycle`. -/
theorem IsCycleOn.isCycle_subtypePerm (hf : f.IsCycleOn s) (hs : s.Nontrivial) :
    (f.subtypePerm fun _ => hf.apply_mem_iff.symm : Perm s).IsCycle := by
  /-
    α : Type u_2
    f : Equiv.Perm α
    s : Set α
    hf : f.IsCycleOn s
    hs : s.Nontrivial
    ⊢ (f.subtypePerm ⋯).IsCycle
  -/
  obtain ⟨a, ha⟩ := hs.nonempty
  exact
    ⟨⟨a, ha⟩, ne_of_apply_ne ((↑) : s → α) (hf.apply_ne hs ha), fun b _ =>
      (hf.2 (⟨a, ha⟩ : s).2 b.2).subtypePerm⟩


/-- Note that the identity is a cycle on any subsingleton set, but not a cycle. -/
protected theorem IsCycleOn.subtypePerm (hf : f.IsCycleOn s) :
    (f.subtypePerm fun _ => hf.apply_mem_iff.symm : Perm s).IsCycleOn _root_.Set.univ := by
  /-
    α : Type u_2
    f : Equiv.Perm α
    s : Set α
    hf : f.IsCycleOn s
    ⊢ (f.subtypePerm ⋯).IsCycleOn _root_.Set.univ
  -/
  obtain hs | hs := s.subsingleton_or_nontrivial
    /-
      case inl
      α : Type u_2
      f : Equiv.Perm α
      s : Set α
      hf : f.IsCycleOn s
      hs : s.Subsingleton
      ⊢ (f.subtypePerm ⋯).IsCycleOn _root_.Set.univ
    -/
  · haveI := hs.coe_sort
    /-
      case inl
      α : Type u_2
      f : Equiv.Perm α
      s : Set α
      hf : f.IsCycleOn s
      hs : s.Subsingleton
      this : Subsingleton ↑s
      ⊢ (f.subtypePerm ⋯).IsCycleOn _root_.Set.univ
    -/
    exact isCycleOn_of_subsingleton _ _
    /-
      🎉 no goals
    -/
  /-
    case inr
    α : Type u_2
    f : Equiv.Perm α
    s : Set α
    hf : f.IsCycleOn s
    hs : s.Nontrivial
    ⊢ (f.subtypePerm ⋯).IsCycleOn _root_.Set.univ
  -/
  convert (hf.isCycle_subtypePerm hs).isCycleOn
  /-
    case h.e'_3
    α : Type u_2
    f : Equiv.Perm α
    s : Set α
    hf : f.IsCycleOn s
    hs : s.Nontrivial
    ⊢ Eq _root_.Set.univ (setOf fun x => Ne ((f.subtypePerm ⋯) x) x)
  -/
  rw [eq_comm, Set.eq_univ_iff_forall]
  /-
    case h.e'_3
    α : Type u_2
    f : Equiv.Perm α
    s : Set α
    hf : f.IsCycleOn s
    hs : s.Nontrivial
    ⊢ ∀ (x : Subtype fun x => Membership.mem s x), Membership.mem (setOf fun x =>  …
  -/
  exact fun x => ne_of_apply_ne ((↑) : s → α) (hf.apply_ne hs x.2)
  /-
    🎉 no goals
  -/

-- TODO: Theory of order of an element under an action

theorem IsCycleOn.pow_apply_eq {s : Finset α} (hf : f.IsCycleOn s) (ha : a ∈ s) {n : ℕ} :
    (f ^ n) a = a ↔ #s ∣ n := by
  /-
    α : Type u_2
    f : Equiv.Perm α
    a : α
    s : Finset α
    hf : f.IsCycleOn ↑s
    ha : Membership.mem s a
    n : Nat
    ⊢ Iff (Eq ((HPow.hPow f n) a) a) (Dvd.dvd s.card n)
  -/
  obtain rfl | hs := Finset.eq_singleton_or_nontrivial ha
    /-
      case inl
      α : Type u_2
      f : Equiv.Perm α
      a : α
      n : Nat
      hf : f.IsCycleOn ↑(Singleton.singleton a)
      ha : Membership.mem (Singleton.singleton a) a
      ⊢ Iff (Eq ((HPow.hPow f n) a) a) (Dvd.dvd (Singleton.singleton a).card n)
    -/
  · rw [coe_singleton, isCycleOn_singleton] at hf
    /-
      case inl
      α : Type u_2
      f : Equiv.Perm α
      a : α
      n : Nat
      hf : Eq (f a) a
      ha : Membership.mem (Singleton.singleton a) a
      ⊢ Iff (Eq ((HPow.hPow f n) a) a) (Dvd.dvd (Singleton.singleton a).card n)
    -/
    simpa using IsFixedPt.iterate hf n
    /-
      🎉 no goals
    -/
  classical
    have h (x : s) : ¬f x = x := hf.apply_ne hs x.2
    have := (hf.isCycle_subtypePerm hs).orderOf
    simp only [coe_sort_coe, support_subtype_perm, ne_eq, h, not_false_eq_true, univ_eq_attach,
      mem_attach, imp_self, implies_true, filter_true_of_mem, card_attach] at this
    rw [← this, orderOf_dvd_iff_pow_eq_one,
      (hf.isCycle_subtypePerm hs).pow_eq_one_iff'
        (ne_of_apply_ne ((↑) : s → α) <| hf.apply_ne hs (⟨a, ha⟩ : s).2)]
    simp
    -- This used to be the end of the proof before https://github.com/leanprover/lean4/pull/2644
    erw [subtypePerm_apply]
    simp


theorem IsCycleOn.zpow_apply_eq {s : Finset α} (hf : f.IsCycleOn s) (ha : a ∈ s) :
    ∀ {n : ℤ}, (f ^ n) a = a ↔ (#s : ℤ) ∣ n
  | Int.ofNat _ => (hf.pow_apply_eq ha).trans Int.natCast_dvd_natCast.symm
  | Int.negSucc n => by
    /-
      α : Type u_2
      f : Equiv.Perm α
      a : α
      s : Finset α
      hf : f.IsCycleOn ↑s
      ha : Membership.mem s a
      n : Nat
      ⊢ Iff (Eq ((HPow.hPow f (Int.negSucc n)) a) a) (Dvd.dvd (↑s.card) (Int.negSucc …
    -/
    rw [zpow_negSucc, ← inv_pow]
    /-
      α : Type u_2
      f : Equiv.Perm α
      a : α
      s : Finset α
      hf : f.IsCycleOn ↑s
      ha : Membership.mem s a
      n : Nat
      ⊢ Iff (Eq ((HPow.hPow (Inv.inv f) (HAdd.hAdd n 1)) a) a) (Dvd.dvd (↑s.card) (I …
    -/
    exact (hf.inv.pow_apply_eq ha).trans (dvd_neg.trans Int.natCast_dvd_natCast).symm
    /-
      🎉 no goals
    -/


theorem IsCycleOn.pow_apply_eq_pow_apply {s : Finset α} (hf : f.IsCycleOn s) (ha : a ∈ s)
    {m n : ℕ} : (f ^ m) a = (f ^ n) a ↔ m ≡ n [MOD #s] := by
  /-
    α : Type u_2
    f : Equiv.Perm α
    a : α
    s : Finset α
    hf : f.IsCycleOn ↑s
    ha : Membership.mem s a
    m n : Nat
    ⊢ Iff (Eq ((HPow.hPow f m) a) ((HPow.hPow f n) a)) (s.card.ModEq m n)
  -/
  rw [Nat.modEq_iff_dvd, ← hf.zpow_apply_eq ha]
  /-
    α : Type u_2
    f : Equiv.Perm α
    a : α
    s : Finset α
    hf : f.IsCycleOn ↑s
    ha : Membership.mem s a
    m n : Nat
    ⊢ Iff (Eq ((HPow.hPow f m) a) ((HPow.hPow f n) a)) (Eq ((HPow.hPow f (HSub.hSu …
  -/
  simp [sub_eq_neg_add, zpow_add, eq_inv_iff_eq, eq_comm]
  /-
    🎉 no goals
  -/


theorem IsCycleOn.zpow_apply_eq_zpow_apply {s : Finset α} (hf : f.IsCycleOn s) (ha : a ∈ s)
    {m n : ℤ} : (f ^ m) a = (f ^ n) a ↔ m ≡ n [ZMOD #s] := by
  /-
    α : Type u_2
    f : Equiv.Perm α
    a : α
    s : Finset α
    hf : f.IsCycleOn ↑s
    ha : Membership.mem s a
    m n : Int
    ⊢ Iff (Eq ((HPow.hPow f m) a) ((HPow.hPow f n) a)) ((↑s.card).ModEq m n)
  -/
  rw [Int.modEq_iff_dvd, ← hf.zpow_apply_eq ha]
  /-
    α : Type u_2
    f : Equiv.Perm α
    a : α
    s : Finset α
    hf : f.IsCycleOn ↑s
    ha : Membership.mem s a
    m n : Int
    ⊢ Iff (Eq ((HPow.hPow f m) a) ((HPow.hPow f n) a)) (Eq ((HPow.hPow f (HSub.hSu …
  -/
  simp [sub_eq_neg_add, zpow_add, eq_inv_iff_eq, eq_comm]
  /-
    🎉 no goals
  -/


theorem IsCycleOn.pow_card_apply {s : Finset α} (hf : f.IsCycleOn s) (ha : a ∈ s) :
    (f ^ #s) a = a :=
  (hf.pow_apply_eq ha).2 dvd_rfl


theorem IsCycleOn.exists_pow_eq {s : Finset α} (hf : f.IsCycleOn s) (ha : a ∈ s) (hb : b ∈ s) :
    ∃ n < #s, (f ^ n) a = b := by
  classical
    obtain ⟨n, rfl⟩ := hf.2 ha hb
    obtain ⟨k, hk⟩ := (Int.mod_modEq n #s).symm.dvd
    refine ⟨n.natMod #s, Int.natMod_lt (Nonempty.card_pos ⟨a, ha⟩).ne', ?_⟩
    rw [← zpow_natCast, Int.natMod,
      Int.toNat_of_nonneg (Int.emod_nonneg _ <| Nat.cast_ne_zero.2
        (Nonempty.card_pos ⟨a, ha⟩).ne'), sub_eq_iff_eq_add'.1 hk, zpow_add, zpow_mul]
    simp only [zpow_natCast, coe_mul, comp_apply, EmbeddingLike.apply_eq_iff_eq]
    exact IsFixedPt.perm_zpow (hf.pow_card_apply ha) _


theorem IsCycleOn.exists_pow_eq' (hs : s.Finite) (hf : f.IsCycleOn s) (ha : a ∈ s) (hb : b ∈ s) :
    ∃ n : ℕ, (f ^ n) a = b := by
  /-
    α : Type u_2
    f : Equiv.Perm α
    s : Set α
    a b : α
    hs : s.Finite
    hf : f.IsCycleOn s
    ha : Membership.mem s a
    hb : Membership.mem s b
    ⊢ Exists fun n => Eq ((HPow.hPow f n) a) b
  -/
  lift s to Finset α using id hs
  /-
    case intro
    α : Type u_2
    f : Equiv.Perm α
    a b : α
    s : Finset α
    hs : (↑s).Finite
    hf : f.IsCycleOn ↑s
    ha : Membership.mem (↑s) a
    hb : Membership.mem (↑s) b
    ⊢ Exists fun n => Eq ((HPow.hPow f n) a) b
  -/
  obtain ⟨n, -, hn⟩ := hf.exists_pow_eq ha hb
  /-
    case intro.intro.intro
    α : Type u_2
    f : Equiv.Perm α
    a b : α
    s : Finset α
    hs : (↑s).Finite
    hf : f.IsCycleOn ↑s
    ha : Membership.mem (↑s) a
    hb : Membership.mem (↑s) b
    n : Nat
    hn : Eq ((HPow.hPow f n) a) b
    ⊢ Exists fun n => Eq ((HPow.hPow f n) a) b
  -/
  exact ⟨n, hn⟩
  /-
    🎉 no goals
  -/


theorem IsCycleOn.range_pow (hs : s.Finite) (h : f.IsCycleOn s) (ha : a ∈ s) :
    Set.range (fun n => (f ^ n) a : ℕ → α) = s :=
  Set.Subset.antisymm (Set.range_subset_iff.2 fun _ => h.1.mapsTo.perm_pow _ ha) fun _ =>
    h.exists_pow_eq' hs ha


theorem IsCycleOn.range_zpow (h : f.IsCycleOn s) (ha : a ∈ s) :
    Set.range (fun n => (f ^ n) a : ℤ → α) = s :=
  Set.Subset.antisymm (Set.range_subset_iff.2 fun _ => (h.1.perm_zpow _).mapsTo ha) <| h.2 ha


theorem IsCycleOn.of_pow {n : ℕ} (hf : (f ^ n).IsCycleOn s) (h : Set.BijOn f s s) : f.IsCycleOn s :=
  ⟨h, fun _ hx _ hy => (hf.2 hx hy).of_pow⟩


theorem IsCycleOn.of_zpow {n : ℤ} (hf : (f ^ n).IsCycleOn s) (h : Set.BijOn f s s) :
    f.IsCycleOn s :=
  ⟨h, fun _ hx _ hy => (hf.2 hx hy).of_zpow⟩


theorem IsCycleOn.extendDomain {p : β → Prop} [DecidablePred p] (f : α ≃ Subtype p)
    (h : g.IsCycleOn s) : (g.extendDomain f).IsCycleOn ((↑) ∘ f '' s) :=
  ⟨h.1.extendDomain, by
    /-
      α : Type u_2
      β : Type u_3
      g : Equiv.Perm α
      s : Set α
      p : β → Prop
      inst✝ : DecidablePred p
      f : Equiv α (Subtype p)
      h : g.IsCycleOn s
      ⊢ ∀ ⦃x : β⦄, Membership.mem (_root_.Set.image (Function.comp Subtype.val ⇑f) s …
    -/
    rintro _ ⟨a, ha, rfl⟩ _ ⟨b, hb, rfl⟩
    /-
      case intro.intro.intro.intro
      α : Type u_2
      β : Type u_3
      g : Equiv.Perm α
      s : Set α
      p : β → Prop
      inst✝ : DecidablePred p
      f : Equiv α (Subtype p)
      h : g.IsCycleOn s
      a : α
      ha : Membership.mem s a
      b : α
      hb : Membership.mem s b
      ⊢ (g.extendDomain f).SameCycle (Function.comp Subtype.val (⇑f) a) (Function.co …
    -/
    exact (h.2 ha hb).extendDomain⟩
    /-
      🎉 no goals
    -/


protected theorem IsCycleOn.countable (hs : f.IsCycleOn s) : s.Countable := by
  /-
    α : Type u_2
    f : Equiv.Perm α
    s : Set α
    hs : f.IsCycleOn s
    ⊢ s.Countable
  -/
  obtain rfl | ⟨a, ha⟩ := s.eq_empty_or_nonempty
    /-
      case inl
      α : Type u_2
      f : Equiv.Perm α
      hs : f.IsCycleOn EmptyCollection.emptyCollection
      ⊢ EmptyCollection.emptyCollection.Countable
    -/
  · exact Set.countable_empty
    /-
      🎉 no goals
    -/
    /-
      case inr.intro
      α : Type u_2
      f : Equiv.Perm α
      s : Set α
      hs : f.IsCycleOn s
      a : α
      ha : Membership.mem s a
      ⊢ s.Countable
    -/
  · exact (Set.countable_range fun n : ℤ => (⇑(f ^ n) : α → α) a).mono (hs.2 ha)
    /-
      🎉 no goals
    -/



theorem Nodup.isCycleOn_formPerm (h : l.Nodup) :
    l.formPerm.IsCycleOn { a | a ∈ l } := by
  /-
    α : Type u_2
    inst✝ : DecidableEq α
    l : List α
    h : l.Nodup
    ⊢ l.formPerm.IsCycleOn (setOf fun a => Membership.mem l a)
  -/
  refine ⟨l.formPerm.bijOn fun _ => List.formPerm_mem_iff_mem, fun a ha b hb => ?_⟩
  /-
    α : Type u_2
    inst✝ : DecidableEq α
    l : List α
    h : l.Nodup
    a : α
    ha : Membership.mem (setOf fun a => Membership.mem l a) a
    b : α
    hb : Membership.mem (setOf fun a => Membership.mem l a) b
    ⊢ l.formPerm.SameCycle a b
  -/
  rw [Set.mem_setOf, ← List.indexOf_lt_length] at ha hb
  /-
    α : Type u_2
    inst✝ : DecidableEq α
    l : List α
    h : l.Nodup
    a : α
    ha : LT.lt (List.indexOf a l) l.length
    b : α
    hb : LT.lt (List.indexOf b l) l.length
    ⊢ l.formPerm.SameCycle a b
  -/
  rw [← List.getElem_indexOf ha, ← List.getElem_indexOf hb]
  /-
    α : Type u_2
    inst✝ : DecidableEq α
    l : List α
    h : l.Nodup
    a : α
    ha : LT.lt (List.indexOf a l) l.length
    b : α
    hb : LT.lt (List.indexOf b l) l.length
    ⊢ l.formPerm.SameCycle (GetElem.getElem l (List.indexOf a l) ha) (GetElem.getE …
  -/
  refine ⟨l.indexOf b - l.indexOf a, ?_⟩
  simp only [sub_eq_neg_add, zpow_add, zpow_neg, Equiv.Perm.inv_eq_iff_eq, zpow_natCast,
    Equiv.Perm.coe_mul, List.formPerm_pow_apply_getElem _ h, Function.comp]
  /-
    α : Type u_2
    inst✝ : DecidableEq α
    l : List α
    h : l.Nodup
    a : α
    ha : LT.lt (List.indexOf a l) l.length
    b : α
    hb : LT.lt (List.indexOf b l) l.length
    ⊢ Eq (GetElem.getElem l (HMod.hMod (HAdd.hAdd (List.indexOf a l) (List.indexOf …
  -/
  rw [add_comm]
  /-
    🎉 no goals
  -/


theorem exists_cycleOn (s : Finset α) :
    ∃ f : Perm α, f.IsCycleOn s ∧ f.support ⊆ s := by
  refine ⟨s.toList.formPerm, ?_, fun x hx => by
    simpa using List.mem_of_formPerm_apply_ne (Perm.mem_support.1 hx)⟩
  /-
    α : Type u_2
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    s : Finset α
    ⊢ s.toList.formPerm.IsCycleOn ↑s
  -/
  convert s.nodup_toList.isCycleOn_formPerm
  /-
    case h.e'_3
    α : Type u_2
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    s : Finset α
    ⊢ Eq (↑s) (setOf fun a => Membership.mem s.toList a)
  -/
  simp
  /-
    🎉 no goals
  -/


theorem Countable.exists_cycleOn (hs : s.Countable) :
    ∃ f : Perm α, f.IsCycleOn s ∧ { x | f x ≠ x } ⊆ s := by
  classical
  obtain hs' | hs' := s.finite_or_infinite
  · refine ⟨hs'.toFinset.toList.formPerm, ?_, fun x hx => by
      simpa using List.mem_of_formPerm_apply_ne hx⟩
    convert hs'.toFinset.nodup_toList.isCycleOn_formPerm
    simp
  · haveI := hs.to_subtype
    haveI := hs'.to_subtype
    obtain ⟨f⟩ : Nonempty (ℤ ≃ s) := inferInstance
    refine ⟨(Equiv.addRight 1).extendDomain f, ?_, fun x hx =>
      of_not_not fun h => hx <| Perm.extendDomain_apply_not_subtype _ _ h⟩
    convert Int.addRight_one_isCycle.isCycleOn.extendDomain f
    rw [Set.image_comp, Equiv.image_eq_preimage]
    ext
    simp


theorem prod_self_eq_iUnion_perm (hf : f.IsCycleOn s) :
    s ×ˢ s = ⋃ n : ℤ, (fun a => (a, (f ^ n) a)) '' s := by
  /-
    α : Type u_2
    f : Equiv.Perm α
    s : Set α
    hf : f.IsCycleOn s
    ⊢ Eq (SProd.sprod s s) (Set.iUnion fun n => Set.image (fun a => { fst := a, sn …
  -/
  ext ⟨a, b⟩
  /-
    case h.mk
    α : Type u_2
    f : Equiv.Perm α
    s : Set α
    hf : f.IsCycleOn s
    a b : α
    ⊢ Iff (Membership.mem (SProd.sprod s s) { fst := a, snd := b }) (Membership.me …
  -/
  simp only [Set.mem_prod, Set.mem_iUnion, Set.mem_image]
  /-
    case h.mk
    α : Type u_2
    f : Equiv.Perm α
    s : Set α
    hf : f.IsCycleOn s
    a b : α
    ⊢ Iff (And (Membership.mem s a) (Membership.mem s b)) (Exists fun i => Exists  …
  -/
  refine ⟨fun hx => ?_, ?_⟩
    /-
      case h.mk.refine_1
      α : Type u_2
      f : Equiv.Perm α
      s : Set α
      hf : f.IsCycleOn s
      a b : α
      hx : And (Membership.mem s a) (Membership.mem s b)
      ⊢ Exists fun i => Exists fun x => And (Membership.mem s x) (Eq { fst := x, snd …
    -/
  · obtain ⟨n, rfl⟩ := hf.2 hx.1 hx.2
    /-
      case h.mk.refine_1.intro
      α : Type u_2
      f : Equiv.Perm α
      s : Set α
      hf : f.IsCycleOn s
      a : α
      n : Int
      hx : And (Membership.mem s a) (Membership.mem s ((HPow.hPow f n) a))
      ⊢ Exists fun i => Exists fun x => And (Membership.mem s x) (Eq { fst := x, snd …
    -/
    exact ⟨_, _, hx.1, rfl⟩
    /-
      🎉 no goals
    -/
    /-
      case h.mk.refine_2
      α : Type u_2
      f : Equiv.Perm α
      s : Set α
      hf : f.IsCycleOn s
      a b : α
      ⊢ (Exists fun i => Exists fun x => And (Membership.mem s x) (Eq { fst := x, sn …
    -/
  · rintro ⟨n, a, ha, ⟨⟩⟩
    /-
      case h.mk.refine_2.intro.intro.intro.refl
      α : Type u_2
      f : Equiv.Perm α
      s : Set α
      hf : f.IsCycleOn s
      a : α
      n : Int
      ha : Membership.mem s a
      ⊢ And (Membership.mem s a) (Membership.mem s ((HPow.hPow f n) a))
    -/
    exact ⟨ha, (hf.1.perm_zpow _).mapsTo ha⟩
    /-
      🎉 no goals
    -/


theorem product_self_eq_disjiUnion_perm_aux (hf : f.IsCycleOn s) :
    (range #s : Set ℕ).PairwiseDisjoint fun k =>
      s.map ⟨fun i => (i, (f ^ k) i), fun _ _ => congr_arg Prod.fst⟩ := by
  /-
    α : Type u_2
    f : Equiv.Perm α
    s : Finset α
    hf : f.IsCycleOn ↑s
    ⊢ (↑(Finset.range s.card)).PairwiseDisjoint fun k => Finset.map { toFun := fun …
  -/
  obtain hs | _ := (s : Set α).subsingleton_or_nontrivial
    /-
      case inl
      α : Type u_2
      f : Equiv.Perm α
      s : Finset α
      hf : f.IsCycleOn ↑s
      hs : (↑s).Subsingleton
      ⊢ (↑(Finset.range s.card)).PairwiseDisjoint fun k => Finset.map { toFun := fun …
    -/
  · refine Set.Subsingleton.pairwise ?_ _
    /-
      case inl
      α : Type u_2
      f : Equiv.Perm α
      s : Finset α
      hf : f.IsCycleOn ↑s
      hs : (↑s).Subsingleton
      ⊢ (↑(Finset.range s.card)).Subsingleton
    -/
    simp_rw [Set.Subsingleton, mem_coe, ← card_le_one] at hs ⊢
    /-
      case inl
      α : Type u_2
      f : Equiv.Perm α
      s : Finset α
      hf : f.IsCycleOn ↑s
      hs : LE.le s.card 1
      ⊢ LE.le (Finset.range s.card).card 1
    -/
    rwa [card_range]
    /-
      🎉 no goals
    -/
  classical
    rintro m hm n hn hmn
    simp only [disjoint_left, Function.onFun, mem_map, Function.Embedding.coeFn_mk, exists_prop,
      not_exists, not_and, forall_exists_index, and_imp, Prod.forall, Prod.mk.inj_iff]
    rintro _ _ _ - rfl rfl a ha rfl h
    rw [hf.pow_apply_eq_pow_apply ha] at h
    rw [mem_coe, mem_range] at hm hn
    exact hmn.symm (h.eq_of_lt_of_lt hn hm)


/-- We can partition the square `s ×ˢ s` into shifted diagonals as such:
```
01234
40123
34012
23401
12340
```

The diagonals are given by the cycle `f`.
-/
theorem product_self_eq_disjiUnion_perm (hf : f.IsCycleOn s) :
    s ×ˢ s =
      (range #s).disjiUnion
        (fun k => s.map ⟨fun i => (i, (f ^ k) i), fun _ _ => congr_arg Prod.fst⟩)
        (product_self_eq_disjiUnion_perm_aux hf) := by
  /-
    α : Type u_2
    f : Equiv.Perm α
    s : Finset α
    hf : f.IsCycleOn ↑s
    ⊢ Eq (SProd.sprod s s) ((Finset.range s.card).disjiUnion (fun k => Finset.map  …
  -/
  ext ⟨a, b⟩
  simp only [mem_product, Equiv.Perm.coe_pow, mem_disjiUnion, mem_range, mem_map,
    Function.Embedding.coeFn_mk, Prod.mk.inj_iff, exists_prop]
  /-
    case h.mk
    α : Type u_2
    f : Equiv.Perm α
    s : Finset α
    hf : f.IsCycleOn ↑s
    a b : α
    ⊢ Iff (And (Membership.mem s a) (Membership.mem s b)) (Exists fun a_1 => And ( …
  -/
  refine ⟨fun hx => ?_, ?_⟩
    /-
      case h.mk.refine_1
      α : Type u_2
      f : Equiv.Perm α
      s : Finset α
      hf : f.IsCycleOn ↑s
      a b : α
      hx : And (Membership.mem s a) (Membership.mem s b)
      ⊢ Exists fun a_1 => And (LT.lt a_1 s.card) (Exists fun a_2 => And (Membership. …
    -/
  · obtain ⟨n, hn, rfl⟩ := hf.exists_pow_eq hx.1 hx.2
    /-
      case h.mk.refine_1.intro.intro
      α : Type u_2
      f : Equiv.Perm α
      s : Finset α
      hf : f.IsCycleOn ↑s
      a : α
      n : Nat
      hn : LT.lt n s.card
      hx : And (Membership.mem s a) (Membership.mem s ((HPow.hPow f n) a))
      ⊢ Exists fun a_1 => And (LT.lt a_1 s.card) (Exists fun a_2 => And (Membership. …
    -/
    exact ⟨n, hn, a, hx.1, rfl, by rw [f.iterate_eq_pow]⟩
    /-
      🎉 no goals
    -/
    /-
      case h.mk.refine_2
      α : Type u_2
      f : Equiv.Perm α
      s : Finset α
      hf : f.IsCycleOn ↑s
      a b : α
      ⊢ (Exists fun a_1 => And (LT.lt a_1 s.card) (Exists fun a_2 => And (Membership …
    -/
  · rintro ⟨n, -, a, ha, rfl, rfl⟩
    /-
      case h.mk.refine_2.intro.intro.intro.intro.intro
      α : Type u_2
      f : Equiv.Perm α
      s : Finset α
      hf : f.IsCycleOn ↑s
      n : Nat
      a : α
      ha : Membership.mem s a
      ⊢ And (Membership.mem s a) (Membership.mem s (Nat.iterate (⇑f) n a))
    -/
    exact ⟨ha, (hf.1.iterate _).mapsTo ha⟩
    /-
      🎉 no goals
    -/


theorem sum_smul_sum_eq_sum_perm (hσ : σ.IsCycleOn s) (f : ι → α) (g : ι → β) :
    (∑ i ∈ s, f i) • ∑ i ∈ s, g i = ∑ k ∈ range #s, ∑ i ∈ s, f i • g ((σ ^ k) i) := by
  /-
    ι : Type u_1
    α : Type u_2
    β : Type u_3
    inst✝² : Semiring α
    inst✝¹ : AddCommMonoid β
    inst✝ : Module α β
    s : Finset ι
    σ : Equiv.Perm ι
    hσ : σ.IsCycleOn ↑s
    f : ι → α
    g : ι → β
    ⊢ Eq (HSMul.hSMul (s.sum fun i => f i) (s.sum fun i => g i)) ((Finset.range s. …
  -/
  rw [sum_smul_sum, ← sum_product']
  /-
    ι : Type u_1
    α : Type u_2
    β : Type u_3
    inst✝² : Semiring α
    inst✝¹ : AddCommMonoid β
    inst✝ : Module α β
    s : Finset ι
    σ : Equiv.Perm ι
    hσ : σ.IsCycleOn ↑s
    f : ι → α
    g : ι → β
    ⊢ Eq ((SProd.sprod s s).sum fun x => HSMul.hSMul (f x.1) (g x.2)) ((Finset.ran …
  -/
  simp_rw [product_self_eq_disjiUnion_perm hσ, sum_disjiUnion, sum_map, Embedding.coeFn_mk]
  /-
    🎉 no goals
  -/


theorem sum_mul_sum_eq_sum_perm (hσ : σ.IsCycleOn s) (f g : ι → α) :
    ((∑ i ∈ s, f i) * ∑ i ∈ s, g i) = ∑ k ∈ range #s, ∑ i ∈ s, f i * g ((σ ^ k) i) :=
  sum_smul_sum_eq_sum_perm hσ f g


theorem subtypePerm_apply_pow_of_mem {g : Perm α} {s : Finset α}
    (hs : ∀ x : α, x ∈ s ↔ g x ∈ s) {n : ℕ} {x : α} (hx : x ∈ s) :
    ((g.subtypePerm hs ^ n) (⟨x, hx⟩ : s) : α) = (g ^ n) x := by
  /-
    α : Type u_2
    g : Equiv.Perm α
    s : Finset α
    hs : ∀ (x : α), Iff (Membership.mem s x) (Membership.mem s (g x))
    n : Nat
    x : α
    hx : Membership.mem s x
    ⊢ Eq (↑((HPow.hPow (g.subtypePerm hs) n) ⟨x, hx⟩)) ((HPow.hPow g n) x)
  -/
  simp only [subtypePerm_pow, subtypePerm_apply]
  /-
    🎉 no goals
  -/


theorem subtypePerm_apply_zpow_of_mem {g : Perm α} {s : Finset α}
    (hs : ∀ x : α, x ∈ s ↔ g x ∈ s) {i : ℤ} {x : α} (hx : x ∈ s) :
    ((g.subtypePerm hs ^ i) (⟨x, hx⟩ : s) : α) = (g ^ i) x := by
  /-
    α : Type u_2
    g : Equiv.Perm α
    s : Finset α
    hs : ∀ (x : α), Iff (Membership.mem s x) (Membership.mem s (g x))
    i : Int
    x : α
    hx : Membership.mem s x
    ⊢ Eq (↑((HPow.hPow (g.subtypePerm hs) i) ⟨x, hx⟩)) ((HPow.hPow g i) x)
  -/
  simp only [subtypePerm_zpow, subtypePerm_apply]
  /-
    🎉 no goals
  -/


/-- Restrict a permutation to its support -/
def subtypePermOfSupport (c : Perm α) : Perm c.support :=
  subtypePerm c fun _ : α => apply_mem_support.symm


/-- Restrict a permutation to a Finset containing its support -/
def subtypePerm_of_support_le (c : Perm α) {s : Finset α}
    (hcs : c.support ⊆ s) : Equiv.Perm s :=
  subtypePerm c (isInvariant_of_support_le hcs)


/-- Support of a cycle is nonempty -/
theorem IsCycle.nonempty_support {g : Perm α} (hg : g.IsCycle) :
    g.support.Nonempty := by
  /-
    α : Type u_2
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    g : Equiv.Perm α
    hg : g.IsCycle
    ⊢ g.support.Nonempty
  -/
  rw [Finset.nonempty_iff_ne_empty, ne_eq, support_eq_empty_iff]
  /-
    α : Type u_2
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    g : Equiv.Perm α
    hg : g.IsCycle
    ⊢ Not (Eq g 1)
  -/
  exact IsCycle.ne_one hg
  /-
    🎉 no goals
  -/


/-- Centralizer of a cycle is a power of that cycle on the cycle -/
theorem IsCycle.commute_iff' {g c : Perm α} (hc : c.IsCycle) :
    Commute g c ↔
      ∃ hc' : ∀ x : α, x ∈ c.support ↔ g x ∈ c.support,
        subtypePerm g hc' ∈ Subgroup.zpowers c.subtypePermOfSupport := by
  /-
    α : Type u_2
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    g c : Equiv.Perm α
    hc : c.IsCycle
    ⊢ Iff (Commute g c) (Exists fun hc' => Membership.mem (Subgroup.zpowers c.subt …
  -/
  constructor
    /-
      case mp
      α : Type u_2
      inst✝¹ : Fintype α
      inst✝ : DecidableEq α
      g c : Equiv.Perm α
      hc : c.IsCycle
      ⊢ Commute g c → Exists fun hc' => Membership.mem (Subgroup.zpowers c.subtypePe …
    -/
  · intro hgc
    /-
      case mp
      α : Type u_2
      inst✝¹ : Fintype α
      inst✝ : DecidableEq α
      g c : Equiv.Perm α
      hc : c.IsCycle
      hgc : Commute g c
      ⊢ Exists fun hc' => Membership.mem (Subgroup.zpowers c.subtypePermOfSupport) ( …
    -/
    have hgc' := mem_support_iff_of_commute hgc
    /-
      case mp
      α : Type u_2
      inst✝¹ : Fintype α
      inst✝ : DecidableEq α
      g c : Equiv.Perm α
      hc : c.IsCycle
      hgc : Commute g c
      hgc' : ∀ (x : α), Iff (Membership.mem c.support x) (Membership.mem c.support ( …
      ⊢ Exists fun hc' => Membership.mem (Subgroup.zpowers c.subtypePermOfSupport) ( …
    -/
    use hgc'
    /-
      case h
      α : Type u_2
      inst✝¹ : Fintype α
      inst✝ : DecidableEq α
      g c : Equiv.Perm α
      hc : c.IsCycle
      hgc : Commute g c
      hgc' : ∀ (x : α), Iff (Membership.mem c.support x) (Membership.mem c.support ( …
      ⊢ Membership.mem (Subgroup.zpowers c.subtypePermOfSupport) (g.subtypePerm hgc')
    -/
    obtain ⟨a, ha⟩ := IsCycle.nonempty_support hc
    /-
      case h.intro
      α : Type u_2
      inst✝¹ : Fintype α
      inst✝ : DecidableEq α
      g c : Equiv.Perm α
      hc : c.IsCycle
      hgc : Commute g c
      hgc' : ∀ (x : α), Iff (Membership.mem c.support x) (Membership.mem c.support ( …
      a : α
      ha : Membership.mem c.support a
      ⊢ Membership.mem (Subgroup.zpowers c.subtypePermOfSupport) (g.subtypePerm hgc')
    -/
    obtain ⟨i, hi⟩ := hc.sameCycle (mem_support.mp ha) (mem_support.mp ((hgc' a).mp ha))
    /-
      case h.intro.intro
      α : Type u_2
      inst✝¹ : Fintype α
      inst✝ : DecidableEq α
      g c : Equiv.Perm α
      hc : c.IsCycle
      hgc : Commute g c
      hgc' : ∀ (x : α), Iff (Membership.mem c.support x) (Membership.mem c.support ( …
      a : α
      ha : Membership.mem c.support a
      i : Int
      hi : Eq ((HPow.hPow c i) a) (g a)
      ⊢ Membership.mem (Subgroup.zpowers c.subtypePermOfSupport) (g.subtypePerm hgc')
    -/
    use i
    /-
      case h
      α : Type u_2
      inst✝¹ : Fintype α
      inst✝ : DecidableEq α
      g c : Equiv.Perm α
      hc : c.IsCycle
      hgc : Commute g c
      hgc' : ∀ (x : α), Iff (Membership.mem c.support x) (Membership.mem c.support ( …
      a : α
      ha : Membership.mem c.support a
      i : Int
      hi : Eq ((HPow.hPow c i) a) (g a)
      ⊢ Eq ((fun x => HPow.hPow c.subtypePermOfSupport x) i) (g.subtypePerm hgc')
    -/
    ext ⟨x, hx⟩
    /-
      case h.H.mk.a
      α : Type u_2
      inst✝¹ : Fintype α
      inst✝ : DecidableEq α
      g c : Equiv.Perm α
      hc : c.IsCycle
      hgc : Commute g c
      hgc' : ∀ (x : α), Iff (Membership.mem c.support x) (Membership.mem c.support ( …
      a : α
      ha : Membership.mem c.support a
      i : Int
      hi : Eq ((HPow.hPow c i) a) (g a)
      x : α
      hx : Membership.mem c.support x
      ⊢ Eq ↑(((fun x => HPow.hPow c.subtypePermOfSupport x) i) ⟨x, hx⟩) ↑((g.subtype …
    -/
    simp only [subtypePermOfSupport, Subtype.coe_mk, subtypePerm_apply]
    /-
      case h.H.mk.a
      α : Type u_2
      inst✝¹ : Fintype α
      inst✝ : DecidableEq α
      g c : Equiv.Perm α
      hc : c.IsCycle
      hgc : Commute g c
      hgc' : ∀ (x : α), Iff (Membership.mem c.support x) (Membership.mem c.support ( …
      a : α
      ha : Membership.mem c.support a
      i : Int
      hi : Eq ((HPow.hPow c i) a) (g a)
      x : α
      hx : Membership.mem c.support x
      ⊢ Eq (↑((HPow.hPow (c.subtypePerm ⋯) i) ⟨x, hx⟩)) (g x)
    -/
    rw [subtypePerm_apply_zpow_of_mem]
    /-
      case h.H.mk.a
      α : Type u_2
      inst✝¹ : Fintype α
      inst✝ : DecidableEq α
      g c : Equiv.Perm α
      hc : c.IsCycle
      hgc : Commute g c
      hgc' : ∀ (x : α), Iff (Membership.mem c.support x) (Membership.mem c.support ( …
      a : α
      ha : Membership.mem c.support a
      i : Int
      hi : Eq ((HPow.hPow c i) a) (g a)
      x : α
      hx : Membership.mem c.support x
      ⊢ Eq ((HPow.hPow c i) x) (g x)
    -/
    obtain ⟨j, rfl⟩ := hc.sameCycle (mem_support.mp ha) (mem_support.mp hx)
    /-
      case h.H.mk.a.intro
      α : Type u_2
      inst✝¹ : Fintype α
      inst✝ : DecidableEq α
      g c : Equiv.Perm α
      hc : c.IsCycle
      hgc : Commute g c
      hgc' : ∀ (x : α), Iff (Membership.mem c.support x) (Membership.mem c.support ( …
      a : α
      ha : Membership.mem c.support a
      i : Int
      hi : Eq ((HPow.hPow c i) a) (g a)
      j : Int
      hx : Membership.mem c.support ((HPow.hPow c j) a)
      ⊢ Eq ((HPow.hPow c i) ((HPow.hPow c j) a)) (g ((HPow.hPow c j) a))
    -/
    simp only [← mul_apply, Commute.eq (Commute.zpow_right hgc j)]
    /-
      case h.H.mk.a.intro
      α : Type u_2
      inst✝¹ : Fintype α
      inst✝ : DecidableEq α
      g c : Equiv.Perm α
      hc : c.IsCycle
      hgc : Commute g c
      hgc' : ∀ (x : α), Iff (Membership.mem c.support x) (Membership.mem c.support ( …
      a : α
      ha : Membership.mem c.support a
      i : Int
      hi : Eq ((HPow.hPow c i) a) (g a)
      j : Int
      hx : Membership.mem c.support ((HPow.hPow c j) a)
      ⊢ Eq ((HMul.hMul (HPow.hPow c i) (HPow.hPow c j)) a) ((HMul.hMul (HPow.hPow c  …
    -/
    rw [← zpow_add, add_comm i j, zpow_add]
    /-
      case h.H.mk.a.intro
      α : Type u_2
      inst✝¹ : Fintype α
      inst✝ : DecidableEq α
      g c : Equiv.Perm α
      hc : c.IsCycle
      hgc : Commute g c
      hgc' : ∀ (x : α), Iff (Membership.mem c.support x) (Membership.mem c.support ( …
      a : α
      ha : Membership.mem c.support a
      i : Int
      hi : Eq ((HPow.hPow c i) a) (g a)
      j : Int
      hx : Membership.mem c.support ((HPow.hPow c j) a)
      ⊢ Eq ((HMul.hMul (HPow.hPow c j) (HPow.hPow c i)) a) ((HMul.hMul (HPow.hPow c  …
    -/
    simp only [mul_apply, EmbeddingLike.apply_eq_iff_eq]
    /-
      case h.H.mk.a.intro
      α : Type u_2
      inst✝¹ : Fintype α
      inst✝ : DecidableEq α
      g c : Equiv.Perm α
      hc : c.IsCycle
      hgc : Commute g c
      hgc' : ∀ (x : α), Iff (Membership.mem c.support x) (Membership.mem c.support ( …
      a : α
      ha : Membership.mem c.support a
      i : Int
      hi : Eq ((HPow.hPow c i) a) (g a)
      j : Int
      hx : Membership.mem c.support ((HPow.hPow c j) a)
      ⊢ Eq ((HPow.hPow c i) a) (g a)
    -/
    exact hi
    /-
      🎉 no goals
    -/
    /-
      case mpr
      α : Type u_2
      inst✝¹ : Fintype α
      inst✝ : DecidableEq α
      g c : Equiv.Perm α
      hc : c.IsCycle
      ⊢ (Exists fun hc' => Membership.mem (Subgroup.zpowers c.subtypePermOfSupport)  …
    -/
  · rintro ⟨hc', ⟨i, hi⟩⟩
    /-
      case mpr.intro.intro
      α : Type u_2
      inst✝¹ : Fintype α
      inst✝ : DecidableEq α
      g c : Equiv.Perm α
      hc : c.IsCycle
      hc' : ∀ (x : α), Iff (Membership.mem c.support x) (Membership.mem c.support (g …
      i : Int
      hi : Eq ((fun x => HPow.hPow c.subtypePermOfSupport x) i) (g.subtypePerm hc')
      ⊢ Commute g c
    -/
    ext x
    /-
      case mpr.intro.intro.H
      α : Type u_2
      inst✝¹ : Fintype α
      inst✝ : DecidableEq α
      g c : Equiv.Perm α
      hc : c.IsCycle
      hc' : ∀ (x : α), Iff (Membership.mem c.support x) (Membership.mem c.support (g …
      i : Int
      hi : Eq ((fun x => HPow.hPow c.subtypePermOfSupport x) i) (g.subtypePerm hc')
      x : α
      ⊢ Eq ((HMul.hMul g c) x) ((HMul.hMul c g) x)
    -/
    simp only [coe_mul, Function.comp_apply]
    /-
      case mpr.intro.intro.H
      α : Type u_2
      inst✝¹ : Fintype α
      inst✝ : DecidableEq α
      g c : Equiv.Perm α
      hc : c.IsCycle
      hc' : ∀ (x : α), Iff (Membership.mem c.support x) (Membership.mem c.support (g …
      i : Int
      hi : Eq ((fun x => HPow.hPow c.subtypePermOfSupport x) i) (g.subtypePerm hc')
      x : α
      ⊢ Eq (g (c x)) (c (g x))
    -/
    by_cases hx : x ∈ c.support
    · suffices hi' : ∀ x ∈ c.support, g x = (c ^ i) x by
        rw [hi' x hx, hi' (c x) (apply_mem_support.mpr hx)]
        simp only [← mul_apply, ← zpow_add_one, ← zpow_one_add, add_comm]
      /-
        case pos
        α : Type u_2
        inst✝¹ : Fintype α
        inst✝ : DecidableEq α
        g c : Equiv.Perm α
        hc : c.IsCycle
        hc' : ∀ (x : α), Iff (Membership.mem c.support x) (Membership.mem c.support (g …
        i : Int
        hi : Eq ((fun x => HPow.hPow c.subtypePermOfSupport x) i) (g.subtypePerm hc')
        x : α
        hx : Membership.mem c.support x
        ⊢ ∀ (x : α), Membership.mem c.support x → Eq (g x) ((HPow.hPow c i) x)
      -/
      intro x hx
      /-
        case pos
        α : Type u_2
        inst✝¹ : Fintype α
        inst✝ : DecidableEq α
        g c : Equiv.Perm α
        hc : c.IsCycle
        hc' : ∀ (x : α), Iff (Membership.mem c.support x) (Membership.mem c.support (g …
        i : Int
        hi : Eq ((fun x => HPow.hPow c.subtypePermOfSupport x) i) (g.subtypePerm hc')
        x✝ : α
        hx✝ : Membership.mem c.support x✝
        x : α
        hx : Membership.mem c.support x
        ⊢ Eq (g x) ((HPow.hPow c i) x)
      -/
      have hix := Perm.congr_fun hi ⟨x, hx⟩
      simp only [← Subtype.coe_inj, subtypePermOfSupport, Subtype.coe_mk, subtypePerm_apply,
        subtypePerm_apply_zpow_of_mem] at hix
      /-
        case pos
        α : Type u_2
        inst✝¹ : Fintype α
        inst✝ : DecidableEq α
        g c : Equiv.Perm α
        hc : c.IsCycle
        hc' : ∀ (x : α), Iff (Membership.mem c.support x) (Membership.mem c.support (g …
        i : Int
        hi : Eq ((fun x => HPow.hPow c.subtypePermOfSupport x) i) (g.subtypePerm hc')
        x✝ : α
        hx✝ : Membership.mem c.support x✝
        x : α
        hx : Membership.mem c.support x
        hix : Eq ((HPow.hPow c i) x) (g x)
        ⊢ Eq (g x) ((HPow.hPow c i) x)
      -/
      exact hix.symm
      /-
        🎉 no goals
      -/
      /-
        case neg
        α : Type u_2
        inst✝¹ : Fintype α
        inst✝ : DecidableEq α
        g c : Equiv.Perm α
        hc : c.IsCycle
        hc' : ∀ (x : α), Iff (Membership.mem c.support x) (Membership.mem c.support (g …
        i : Int
        hi : Eq ((fun x => HPow.hPow c.subtypePermOfSupport x) i) (g.subtypePerm hc')
        x : α
        hx : Not (Membership.mem c.support x)
        ⊢ Eq (g (c x)) (c (g x))
      -/
    · rw [not_mem_support.mp hx, eq_comm, ← not_mem_support]
      /-
        case neg
        α : Type u_2
        inst✝¹ : Fintype α
        inst✝ : DecidableEq α
        g c : Equiv.Perm α
        hc : c.IsCycle
        hc' : ∀ (x : α), Iff (Membership.mem c.support x) (Membership.mem c.support (g …
        i : Int
        hi : Eq ((fun x => HPow.hPow c.subtypePermOfSupport x) i) (g.subtypePerm hc')
        x : α
        hx : Not (Membership.mem c.support x)
        ⊢ Not (Membership.mem c.support (g x))
      -/
      contrapose! hx
      /-
        case neg
        α : Type u_2
        inst✝¹ : Fintype α
        inst✝ : DecidableEq α
        g c : Equiv.Perm α
        hc : c.IsCycle
        hc' : ∀ (x : α), Iff (Membership.mem c.support x) (Membership.mem c.support (g …
        i : Int
        hi : Eq ((fun x => HPow.hPow c.subtypePermOfSupport x) i) (g.subtypePerm hc')
        x : α
        hx : Membership.mem c.support (g x)
        ⊢ Membership.mem c.support x
      -/
      exact (hc' x).mpr hx
      /-
        🎉 no goals
      -/


/-- A permutation `g` commutes with a cycle `c` if and only if
  `c.support` is invariant under `g`, and `g` acts on it as a power of `c`. -/
theorem IsCycle.commute_iff {g c : Perm α} (hc : c.IsCycle) :
    Commute g c ↔
      ∃ hc' : ∀ x : α, x ∈ c.support ↔ g x ∈ c.support,
        ofSubtype (subtypePerm g hc') ∈ Subgroup.zpowers c := by
  /-
    α : Type u_2
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    g c : Equiv.Perm α
    hc : c.IsCycle
    ⊢ Iff (Commute g c) (Exists fun hc' => Membership.mem (Subgroup.zpowers c) (Eq …
  -/
  simp_rw [hc.commute_iff', Subgroup.mem_zpowers_iff]
  /-
    α : Type u_2
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    g c : Equiv.Perm α
    hc : c.IsCycle
    ⊢ Iff (Exists fun h => Exists fun k => Eq (HPow.hPow c.subtypePermOfSupport k) …
  -/
  refine exists_congr fun hc' => exists_congr fun k => ?_
  /-
    α : Type u_2
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    g c : Equiv.Perm α
    hc : c.IsCycle
    hc' : ∀ (x : α), Iff (Membership.mem c.support x) (Membership.mem c.support (g …
    k : Int
    ⊢ Iff (Eq (HPow.hPow c.subtypePermOfSupport k) (g.subtypePerm ⋯)) (Eq (HPow.hP …
  -/
  rw [subtypePermOfSupport, subtypePerm_zpow c k]
  /-
    α : Type u_2
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    g c : Equiv.Perm α
    hc : c.IsCycle
    hc' : ∀ (x : α), Iff (Membership.mem c.support x) (Membership.mem c.support (g …
    k : Int
    ⊢ Iff (Eq ((HPow.hPow c k).subtypePerm ⋯) (g.subtypePerm ⋯)) (Eq (HPow.hPow c  …
  -/
  simp only [Perm.ext_iff, subtypePerm_apply, Subtype.mk.injEq, Subtype.forall]
  /-
    α : Type u_2
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    g c : Equiv.Perm α
    hc : c.IsCycle
    hc' : ∀ (x : α), Iff (Membership.mem c.support x) (Membership.mem c.support (g …
    k : Int
    ⊢ Iff (∀ (a : α), Membership.mem c.support a → Eq ((HPow.hPow c k) a) (g a)) ( …
  -/
  apply forall_congr'
  /-
    case h
    α : Type u_2
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    g c : Equiv.Perm α
    hc : c.IsCycle
    hc' : ∀ (x : α), Iff (Membership.mem c.support x) (Membership.mem c.support (g …
    k : Int
    ⊢ ∀ (a : α), Iff (Membership.mem c.support a → Eq ((HPow.hPow c k) a) (g a)) ( …
  -/
  intro a
  /-
    case h
    α : Type u_2
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    g c : Equiv.Perm α
    hc : c.IsCycle
    hc' : ∀ (x : α), Iff (Membership.mem c.support x) (Membership.mem c.support (g …
    k : Int
    a : α
    ⊢ Iff (Membership.mem c.support a → Eq ((HPow.hPow c k) a) (g a)) (Eq ((HPow.h …
  -/
  by_cases ha : a ∈ c.support
    /-
      case pos
      α : Type u_2
      inst✝¹ : Fintype α
      inst✝ : DecidableEq α
      g c : Equiv.Perm α
      hc : c.IsCycle
      hc' : ∀ (x : α), Iff (Membership.mem c.support x) (Membership.mem c.support (g …
      k : Int
      a : α
      ha : Membership.mem c.support a
      ⊢ Iff (Membership.mem c.support a → Eq ((HPow.hPow c k) a) (g a)) (Eq ((HPow.h …
    -/
  · rw [imp_iff_right ha, ofSubtype_subtypePerm_of_mem hc' ha]
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_2
      inst✝¹ : Fintype α
      inst✝ : DecidableEq α
      g c : Equiv.Perm α
      hc : c.IsCycle
      hc' : ∀ (x : α), Iff (Membership.mem c.support x) (Membership.mem c.support (g …
      k : Int
      a : α
      ha : Not (Membership.mem c.support a)
      ⊢ Iff (Membership.mem c.support a → Eq ((HPow.hPow c k) a) (g a)) (Eq ((HPow.h …
    -/
  · rw [iff_true_left (fun b ↦ (ha b).elim), ofSubtype_apply_of_not_mem, ← not_mem_support]
      /-
        case neg
        α : Type u_2
        inst✝¹ : Fintype α
        inst✝ : DecidableEq α
        g c : Equiv.Perm α
        hc : c.IsCycle
        hc' : ∀ (x : α), Iff (Membership.mem c.support x) (Membership.mem c.support (g …
        k : Int
        a : α
        ha : Not (Membership.mem c.support a)
        ⊢ Not (Membership.mem (HPow.hPow c k).support a)
      -/
    · exact Finset.not_mem_mono (support_zpow_le c k) ha
      /-
        🎉 no goals
      -/
      /-
        case neg.ha
        α : Type u_2
        inst✝¹ : Fintype α
        inst✝ : DecidableEq α
        g c : Equiv.Perm α
        hc : c.IsCycle
        hc' : ∀ (x : α), Iff (Membership.mem c.support x) (Membership.mem c.support (g …
        k : Int
        a : α
        ha : Not (Membership.mem c.support a)
        ⊢ Not (Membership.mem c.support a)
      -/
    · exact ha
      /-
        🎉 no goals
      -/


theorem zpow_eq_ofSubtype_subtypePerm_iff
    {g c : Equiv.Perm α} {s : Finset α}
    (hg : ∀ x, x ∈ s ↔ g x ∈ s) (hc : c.support ⊆ s) (n : ℤ) :
    c ^ n = ofSubtype (g.subtypePerm hg) ↔
      c.subtypePerm (isInvariant_of_support_le hc) ^ n = g.subtypePerm hg := by
  /-
    α : Type u_2
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    g c : Equiv.Perm α
    s : Finset α
    hg : ∀ (x : α), Iff (Membership.mem s x) (Membership.mem s (g x))
    hc : HasSubset.Subset c.support s
    n : Int
    ⊢ Iff (Eq (HPow.hPow c n) (Equiv.Perm.ofSubtype (g.subtypePerm hg))) (Eq (HPow …
  -/
  constructor
    /-
      case mp
      α : Type u_2
      inst✝¹ : Fintype α
      inst✝ : DecidableEq α
      g c : Equiv.Perm α
      s : Finset α
      hg : ∀ (x : α), Iff (Membership.mem s x) (Membership.mem s (g x))
      hc : HasSubset.Subset c.support s
      n : Int
      ⊢ Eq (HPow.hPow c n) (Equiv.Perm.ofSubtype (g.subtypePerm hg)) → Eq (HPow.hPow …
    -/
  · intro h
    /-
      case mp
      α : Type u_2
      inst✝¹ : Fintype α
      inst✝ : DecidableEq α
      g c : Equiv.Perm α
      s : Finset α
      hg : ∀ (x : α), Iff (Membership.mem s x) (Membership.mem s (g x))
      hc : HasSubset.Subset c.support s
      n : Int
      h : Eq (HPow.hPow c n) (Equiv.Perm.ofSubtype (g.subtypePerm hg))
      ⊢ Eq (HPow.hPow (c.subtypePerm ⋯) n) (g.subtypePerm hg)
    -/
    ext ⟨x, hx⟩
    /-
      case mp.H.mk.a
      α : Type u_2
      inst✝¹ : Fintype α
      inst✝ : DecidableEq α
      g c : Equiv.Perm α
      s : Finset α
      hg : ∀ (x : α), Iff (Membership.mem s x) (Membership.mem s (g x))
      hc : HasSubset.Subset c.support s
      n : Int
      h : Eq (HPow.hPow c n) (Equiv.Perm.ofSubtype (g.subtypePerm hg))
      x : α
      hx : Membership.mem s x
      ⊢ Eq ↑((HPow.hPow (c.subtypePerm ⋯) n) ⟨x, hx⟩) ↑((g.subtypePerm hg) ⟨x, hx⟩)
    -/
    simp only [Perm.congr_fun h x, subtypePerm_apply_zpow_of_mem, Subtype.coe_mk, subtypePerm_apply]
    /-
      case mp.H.mk.a
      α : Type u_2
      inst✝¹ : Fintype α
      inst✝ : DecidableEq α
      g c : Equiv.Perm α
      s : Finset α
      hg : ∀ (x : α), Iff (Membership.mem s x) (Membership.mem s (g x))
      hc : HasSubset.Subset c.support s
      n : Int
      h : Eq (HPow.hPow c n) (Equiv.Perm.ofSubtype (g.subtypePerm hg))
      x : α
      hx : Membership.mem s x
      ⊢ Eq ((Equiv.Perm.ofSubtype (g.subtypePerm hg)) x) (g x)
    -/
    rw [ofSubtype_apply_of_mem]
      /-
        case mp.H.mk.a
        α : Type u_2
        inst✝¹ : Fintype α
        inst✝ : DecidableEq α
        g c : Equiv.Perm α
        s : Finset α
        hg : ∀ (x : α), Iff (Membership.mem s x) (Membership.mem s (g x))
        hc : HasSubset.Subset c.support s
        n : Int
        h : Eq (HPow.hPow c n) (Equiv.Perm.ofSubtype (g.subtypePerm hg))
        x : α
        hx : Membership.mem s x
        ⊢ Eq (↑((g.subtypePerm hg) ⟨x, ?mp.H.mk.a.ha⟩)) (g x)
      -/
    · simp only [Subtype.coe_mk, subtypePerm_apply]
      /-
        🎉 no goals
      -/
      /-
        case mp.H.mk.a.ha
        α : Type u_2
        inst✝¹ : Fintype α
        inst✝ : DecidableEq α
        g c : Equiv.Perm α
        s : Finset α
        hg : ∀ (x : α), Iff (Membership.mem s x) (Membership.mem s (g x))
        hc : HasSubset.Subset c.support s
        n : Int
        h : Eq (HPow.hPow c n) (Equiv.Perm.ofSubtype (g.subtypePerm hg))
        x : α
        hx : Membership.mem s x
        ⊢ Membership.mem s x
      -/
    · exact hx
      /-
        🎉 no goals
      -/
    /-
      case mpr
      α : Type u_2
      inst✝¹ : Fintype α
      inst✝ : DecidableEq α
      g c : Equiv.Perm α
      s : Finset α
      hg : ∀ (x : α), Iff (Membership.mem s x) (Membership.mem s (g x))
      hc : HasSubset.Subset c.support s
      n : Int
      ⊢ Eq (HPow.hPow (c.subtypePerm ⋯) n) (g.subtypePerm hg) → Eq (HPow.hPow c n) ( …
    -/
  · intro h; ext x
    /-
      case mpr.H
      α : Type u_2
      inst✝¹ : Fintype α
      inst✝ : DecidableEq α
      g c : Equiv.Perm α
      s : Finset α
      hg : ∀ (x : α), Iff (Membership.mem s x) (Membership.mem s (g x))
      hc : HasSubset.Subset c.support s
      n : Int
      h : Eq (HPow.hPow (c.subtypePerm ⋯) n) (g.subtypePerm hg)
      x : α
      ⊢ Eq ((HPow.hPow c n) x) ((Equiv.Perm.ofSubtype (g.subtypePerm hg)) x)
    -/
    rw [← h]
    /-
      case mpr.H
      α : Type u_2
      inst✝¹ : Fintype α
      inst✝ : DecidableEq α
      g c : Equiv.Perm α
      s : Finset α
      hg : ∀ (x : α), Iff (Membership.mem s x) (Membership.mem s (g x))
      hc : HasSubset.Subset c.support s
      n : Int
      h : Eq (HPow.hPow (c.subtypePerm ⋯) n) (g.subtypePerm hg)
      x : α
      ⊢ Eq ((HPow.hPow c n) x) ((Equiv.Perm.ofSubtype (HPow.hPow (c.subtypePerm ⋯) n …
    -/
    by_cases hx : x ∈ s
    · rw [ofSubtype_apply_of_mem (subtypePerm c _ ^ n) hx,
        subtypePerm_zpow, subtypePerm_apply]
    · rw [ofSubtype_apply_of_not_mem (subtypePerm c _ ^ n) hx,
        ← not_mem_support]
      /-
        case neg
        α : Type u_2
        inst✝¹ : Fintype α
        inst✝ : DecidableEq α
        g c : Equiv.Perm α
        s : Finset α
        hg : ∀ (x : α), Iff (Membership.mem s x) (Membership.mem s (g x))
        hc : HasSubset.Subset c.support s
        n : Int
        h : Eq (HPow.hPow (c.subtypePerm ⋯) n) (g.subtypePerm hg)
        x : α
        hx : Not (Membership.mem s x)
        ⊢ Not (Membership.mem (HPow.hPow c n).support x)
      -/
      exact fun hx' ↦ hx (hc (support_zpow_le _ _ hx'))
      /-
        🎉 no goals
      -/


theorem cycle_zpow_mem_support_iff {g : Perm α}
    (hg : g.IsCycle) {n : ℤ} {x : α} (hx : g x ≠ x) :
    (g ^ n) x = x ↔ n % #g.support = 0 := by
  /-
    α : Type u_2
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    g : Equiv.Perm α
    hg : g.IsCycle
    n : Int
    x : α
    hx : Ne (g x) x
    ⊢ Iff (Eq ((HPow.hPow g n) x) x) (Eq (HMod.hMod n ↑g.support.card) 0)
  -/
  set q := n / #g.support
  /-
    α : Type u_2
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    g : Equiv.Perm α
    hg : g.IsCycle
    n : Int
    x : α
    hx : Ne (g x) x
    q : Int := HDiv.hDiv n ↑g.support.card
    ⊢ Iff (Eq ((HPow.hPow g n) x) x) (Eq (HMod.hMod n ↑g.support.card) 0)
  -/
  set r := n % #g.support
  have div_euc : r + #g.support * q = n ∧ 0 ≤ r ∧ r < #g.support := by
    rw [← Int.ediv_emod_unique _]
    · exact ⟨rfl, rfl⟩
    simp only [Int.natCast_pos]
    apply lt_of_lt_of_le _ (IsCycle.two_le_card_support hg); norm_num
  /-
    α : Type u_2
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    g : Equiv.Perm α
    hg : g.IsCycle
    n : Int
    x : α
    hx : Ne (g x) x
    q : Int := HDiv.hDiv n ↑g.support.card
    r : Int := HMod.hMod n ↑g.support.card
    div_euc : And (Eq (HAdd.hAdd r (HMul.hMul (↑g.support.card) q)) n) (And (LE.le …
    ⊢ Iff (Eq ((HPow.hPow g n) x) x) (Eq r 0)
  -/
  simp only [← hg.orderOf] at div_euc
  /-
    α : Type u_2
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    g : Equiv.Perm α
    hg : g.IsCycle
    n : Int
    x : α
    hx : Ne (g x) x
    q : Int := HDiv.hDiv n ↑g.support.card
    r : Int := HMod.hMod n ↑g.support.card
    div_euc : And (Eq (HAdd.hAdd r (HMul.hMul (↑(orderOf g)) q)) n) (And (LE.le 0  …
    ⊢ Iff (Eq ((HPow.hPow g n) x) x) (Eq r 0)
  -/
  obtain ⟨m, hm⟩ := Int.eq_ofNat_of_zero_le div_euc.2.1
  /-
    case intro
    α : Type u_2
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    g : Equiv.Perm α
    hg : g.IsCycle
    n : Int
    x : α
    hx : Ne (g x) x
    q : Int := HDiv.hDiv n ↑g.support.card
    r : Int := HMod.hMod n ↑g.support.card
    div_euc : And (Eq (HAdd.hAdd r (HMul.hMul (↑(orderOf g)) q)) n) (And (LE.le 0  …
    m : Nat
    hm : Eq r ↑m
    ⊢ Iff (Eq ((HPow.hPow g n) x) x) (Eq r 0)
  -/
  simp only [hm, Nat.cast_nonneg, Nat.cast_lt, true_and] at div_euc
  /-
    case intro
    α : Type u_2
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    g : Equiv.Perm α
    hg : g.IsCycle
    n : Int
    x : α
    hx : Ne (g x) x
    q : Int := HDiv.hDiv n ↑g.support.card
    r : Int := HMod.hMod n ↑g.support.card
    m : Nat
    hm : Eq r ↑m
    div_euc : And (Eq (HAdd.hAdd (↑m) (HMul.hMul (↑(orderOf g)) q)) n) (LT.lt m (o …
    ⊢ Iff (Eq ((HPow.hPow g n) x) x) (Eq r 0)
  -/
  rw [← div_euc.1, zpow_add g]
  simp only [hm, Nat.cast_eq_zero, zpow_natCast, coe_mul, comp_apply,zpow_mul,
    pow_orderOf_eq_one, one_zpow, coe_one, id_eq]
  have : (g ^ m) x = x ↔ g ^ m = 1 := by
    constructor
    · intro hgm
      simp only [IsCycle.pow_eq_one_iff hg]
      use x
    · intro hgm
      simp only [hgm, coe_one, id_eq]
  /-
    case intro
    α : Type u_2
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    g : Equiv.Perm α
    hg : g.IsCycle
    n : Int
    x : α
    hx : Ne (g x) x
    q : Int := HDiv.hDiv n ↑g.support.card
    r : Int := HMod.hMod n ↑g.support.card
    m : Nat
    hm : Eq r ↑m
    div_euc : And (Eq (HAdd.hAdd (↑m) (HMul.hMul (↑(orderOf g)) q)) n) (LT.lt m (o …
    this : Iff (Eq ((HPow.hPow g m) x) x) (Eq (HPow.hPow g m) 1)
    ⊢ Iff (Eq ((HPow.hPow g m) x) x) (Eq m 0)
  -/
  rw [this]
  /-
    case intro
    α : Type u_2
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    g : Equiv.Perm α
    hg : g.IsCycle
    n : Int
    x : α
    hx : Ne (g x) x
    q : Int := HDiv.hDiv n ↑g.support.card
    r : Int := HMod.hMod n ↑g.support.card
    m : Nat
    hm : Eq r ↑m
    div_euc : And (Eq (HAdd.hAdd (↑m) (HMul.hMul (↑(orderOf g)) q)) n) (LT.lt m (o …
    this : Iff (Eq ((HPow.hPow g m) x) x) (Eq (HPow.hPow g m) 1)
    ⊢ Iff (Eq (HPow.hPow g m) 1) (Eq m 0)
  -/
  by_cases hm0 : m = 0
    /-
      case pos
      α : Type u_2
      inst✝¹ : Fintype α
      inst✝ : DecidableEq α
      g : Equiv.Perm α
      hg : g.IsCycle
      n : Int
      x : α
      hx : Ne (g x) x
      q : Int := HDiv.hDiv n ↑g.support.card
      r : Int := HMod.hMod n ↑g.support.card
      m : Nat
      hm : Eq r ↑m
      div_euc : And (Eq (HAdd.hAdd (↑m) (HMul.hMul (↑(orderOf g)) q)) n) (LT.lt m (o …
      this : Iff (Eq ((HPow.hPow g m) x) x) (Eq (HPow.hPow g m) 1)
      hm0 : Eq m 0
      ⊢ Iff (Eq (HPow.hPow g m) 1) (Eq m 0)
    -/
  · simp only [hm0, pow_zero, Nat.cast_zero]
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_2
      inst✝¹ : Fintype α
      inst✝ : DecidableEq α
      g : Equiv.Perm α
      hg : g.IsCycle
      n : Int
      x : α
      hx : Ne (g x) x
      q : Int := HDiv.hDiv n ↑g.support.card
      r : Int := HMod.hMod n ↑g.support.card
      m : Nat
      hm : Eq r ↑m
      div_euc : And (Eq (HAdd.hAdd (↑m) (HMul.hMul (↑(orderOf g)) q)) n) (LT.lt m (o …
      this : Iff (Eq ((HPow.hPow g m) x) x) (Eq (HPow.hPow g m) 1)
      hm0 : Not (Eq m 0)
      ⊢ Iff (Eq (HPow.hPow g m) 1) (Eq m 0)
    -/
  · simp only [Nat.cast_eq_zero, hm0, iff_false]
    /-
      case neg
      α : Type u_2
      inst✝¹ : Fintype α
      inst✝ : DecidableEq α
      g : Equiv.Perm α
      hg : g.IsCycle
      n : Int
      x : α
      hx : Ne (g x) x
      q : Int := HDiv.hDiv n ↑g.support.card
      r : Int := HMod.hMod n ↑g.support.card
      m : Nat
      hm : Eq r ↑m
      div_euc : And (Eq (HAdd.hAdd (↑m) (HMul.hMul (↑(orderOf g)) q)) n) (LT.lt m (o …
      this : Iff (Eq ((HPow.hPow g m) x) x) (Eq (HPow.hPow g m) 1)
      hm0 : Not (Eq m 0)
      ⊢ Not (Eq (HPow.hPow g m) 1)
    -/
    exact pow_ne_one_of_lt_orderOf hm0 div_euc.2
    /-
      🎉 no goals
    -/


