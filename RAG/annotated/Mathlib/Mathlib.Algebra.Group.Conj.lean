/-- We say that `a` is conjugate to `b` if for some unit `c` we have `c * a * c⁻¹ = b`. -/
def IsConj (a b : α) :=
  ∃ c : αˣ, SemiconjBy (↑c) a b


@[refl]
theorem IsConj.refl (a : α) : IsConj a a :=
  ⟨1, SemiconjBy.one_left a⟩


@[symm]
theorem IsConj.symm {a b : α} : IsConj a b → IsConj b a
  | ⟨c, hc⟩ => ⟨c⁻¹, hc.units_inv_symm_left⟩


theorem isConj_comm {g h : α} : IsConj g h ↔ IsConj h g :=
  ⟨IsConj.symm, IsConj.symm⟩


@[trans]
theorem IsConj.trans {a b c : α} : IsConj a b → IsConj b c → IsConj a c
  | ⟨c₁, hc₁⟩, ⟨c₂, hc₂⟩ => ⟨c₂ * c₁, hc₂.mul_left hc₁⟩


@[simp]
theorem isConj_iff_eq {α : Type*} [CommMonoid α] {a b : α} : IsConj a b ↔ a = b :=
  ⟨fun ⟨c, hc⟩ => by
    /-
      α : Type u_1
      inst✝ : CommMonoid α
      a b : α
      x✝ : IsConj a b
      c : Units α
      hc : SemiconjBy (↑c) a b
      ⊢ Eq a b
    -/
    rw [SemiconjBy, mul_comm, ← Units.mul_inv_eq_iff_eq_mul, mul_assoc, c.mul_inv, mul_one] at hc
    /-
      α : Type u_1
      inst✝ : CommMonoid α
      a b : α
      x✝ : IsConj a b
      c : Units α
      hc : Eq a b
      ⊢ Eq a b
    -/
    /-
      🎉 no goals
    -/
    exact hc, fun h => by rw [h]⟩
                          /-
                            🎉 no goals
                          -/


protected theorem MonoidHom.map_isConj (f : α →* β) {a b : α} : IsConj a b → IsConj (f a) (f b)
                                  /-
                                    α : Type u
                                    β : Type v
                                    inst✝¹ : Monoid α
                                    inst✝ : Monoid β
                                    f : MonoidHom α β
                                    a b : α
                                    c : Units α
                                    hc : SemiconjBy (↑c) a b
                                    ⊢ SemiconjBy (↑((Units.map f) c)) (f a) (f b)
                                  -/
  | ⟨c, hc⟩ => ⟨Units.map f c, by rw [Units.coe_map, SemiconjBy, ← f.map_mul, hc.eq, f.map_mul]⟩
                                  /-
                                    🎉 no goals
                                  -/


@[simp]
theorem isConj_one_right {a : α} : IsConj 1 a ↔ a = 1 :=
  ⟨fun ⟨_, hc⟩ => mul_right_cancel (hc.symm.trans ((mul_one _).trans (one_mul _).symm)), fun h => by
    /-
      α : Type u
      inst✝ : CancelMonoid α
      a : α
      h : Eq a 1
      ⊢ IsConj 1 a
    -/
    rw [h]⟩
    /-
      🎉 no goals
    -/


@[simp]
theorem isConj_one_left {a : α} : IsConj a 1 ↔ a = 1 :=
  calc
    IsConj a 1 ↔ IsConj 1 a := ⟨IsConj.symm, IsConj.symm⟩
    _ ↔ a = 1 := isConj_one_right


@[simp]
theorem isConj_iff {a b : α} : IsConj a b ↔ ∃ c : α, c * a * c⁻¹ = b :=
  ⟨fun ⟨c, hc⟩ => ⟨c, mul_inv_eq_iff_eq_mul.2 hc⟩, fun ⟨c, hc⟩ =>
    ⟨⟨c, c⁻¹, mul_inv_cancel c, inv_mul_cancel c⟩, mul_inv_eq_iff_eq_mul.1 hc⟩⟩

-- Porting note: not in simp NF.
-- @[simp]

theorem conj_inv {a b : α} : (b * a * b⁻¹)⁻¹ = b * a⁻¹ * b⁻¹ :=
  (map_inv (MulAut.conj b) a).symm


@[simp]
theorem conj_mul {a b c : α} : b * a * b⁻¹ * (b * c * b⁻¹) = b * (a * c) * b⁻¹ :=
  (map_mul (MulAut.conj b) a c).symm


@[simp]
theorem conj_pow {i : ℕ} {a b : α} : (a * b * a⁻¹) ^ i = a * b ^ i * a⁻¹ := by
  /-
    α : Type u
    inst✝ : Group α
    i : Nat
    a b : α
    ⊢ Eq (HPow.hPow (HMul.hMul (HMul.hMul a b) (Inv.inv a)) i) (HMul.hMul (HMul.hM …
  -/
  induction' i with i hi
    /-
      case zero
      α : Type u
      inst✝ : Group α
      a b : α
      ⊢ Eq (HPow.hPow (HMul.hMul (HMul.hMul a b) (Inv.inv a)) 0) (HMul.hMul (HMul.hM …
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case succ
      α : Type u
      inst✝ : Group α
      a b : α
      i : Nat
      hi : Eq (HPow.hPow (HMul.hMul (HMul.hMul a b) (Inv.inv a)) i) (HMul.hMul (HMul …
      ⊢ Eq (HPow.hPow (HMul.hMul (HMul.hMul a b) (Inv.inv a)) (HAdd.hAdd i 1)) (HMul …
    -/
  · simp [pow_succ, hi]
    /-
      🎉 no goals
    -/


@[simp]
theorem conj_zpow {i : ℤ} {a b : α} : (a * b * a⁻¹) ^ i = a * b ^ i * a⁻¹ := by
  /-
    α : Type u
    inst✝ : Group α
    i : Int
    a b : α
    ⊢ Eq (HPow.hPow (HMul.hMul (HMul.hMul a b) (Inv.inv a)) i) (HMul.hMul (HMul.hM …
  -/
  induction i
    /-
      case ofNat
      α : Type u
      inst✝ : Group α
      a b : α
      a✝ : Nat
      ⊢ Eq (HPow.hPow (HMul.hMul (HMul.hMul a b) (Inv.inv a)) (Int.ofNat a✝)) (HMul. …
    -/
  · change (a * b * a⁻¹) ^ (_ : ℤ) = a * b ^ (_ : ℤ) * a⁻¹
    /-
      case ofNat
      α : Type u
      inst✝ : Group α
      a b : α
      a✝ : Nat
      ⊢ Eq (HPow.hPow (HMul.hMul (HMul.hMul a b) (Inv.inv a)) (Int.ofNat a✝)) (HMul. …
    -/
    simp [zpow_natCast]
    /-
      🎉 no goals
    -/
    /-
      case negSucc
      α : Type u
      inst✝ : Group α
      a b : α
      a✝ : Nat
      ⊢ Eq (HPow.hPow (HMul.hMul (HMul.hMul a b) (Inv.inv a)) (Int.negSucc a✝)) (HMu …
    -/
  · simp only [zpow_negSucc, conj_pow, mul_inv_rev, inv_inv]
    /-
      case negSucc
      α : Type u
      inst✝ : Group α
      a b : α
      a✝ : Nat
      ⊢ Eq (HMul.hMul a (HMul.hMul (Inv.inv (HPow.hPow b (HAdd.hAdd a✝ 1))) (Inv.inv …
    -/
    rw [mul_assoc]
    /-
      🎉 no goals
    -/
-- Porting note: Added `change`, `zpow_natCast`, and `rw`.


theorem conj_injective {x : α} : Function.Injective fun g : α => x * g * x⁻¹ :=
  (MulAut.conj x).injective


/-- The setoid of the relation `IsConj` iff there is a unit `u` such that `u * x = y * u` -/
protected def setoid (α : Type*) [Monoid α] : Setoid α where
  r := IsConj
  iseqv := ⟨IsConj.refl, IsConj.symm, IsConj.trans⟩


/-- The quotient type of conjugacy classes of a group. -/
def ConjClasses (α : Type*) [Monoid α] : Type _ :=
  Quotient (IsConj.setoid α)


/-- The canonical quotient map from a monoid `α` into the `ConjClasses` of `α` -/
protected def mk {α : Type*} [Monoid α] (a : α) : ConjClasses α := ⟦a⟧


instance : Inhabited (ConjClasses α) := ⟨⟦1⟧⟩


theorem mk_eq_mk_iff_isConj {a b : α} : ConjClasses.mk a = ConjClasses.mk b ↔ IsConj a b :=
  Iff.intro Quotient.exact Quot.sound


theorem quotient_mk_eq_mk (a : α) : ⟦a⟧ = ConjClasses.mk a :=
  rfl


theorem quot_mk_eq_mk (a : α) : Quot.mk Setoid.r a = ConjClasses.mk a :=
  rfl


theorem forall_isConj {p : ConjClasses α → Prop} : (∀ a, p a) ↔ ∀ a, p (ConjClasses.mk a) :=
  Iff.intro (fun h _ => h _) fun h a => Quotient.inductionOn a h


theorem mk_surjective : Function.Surjective (@ConjClasses.mk α _) :=
  forall_isConj.2 fun a => ⟨a, rfl⟩


instance : One (ConjClasses α) :=
  ⟨⟦1⟧⟩


theorem one_eq_mk_one : (1 : ConjClasses α) = ConjClasses.mk 1 :=
  rfl


theorem exists_rep (a : ConjClasses α) : ∃ a0 : α, ConjClasses.mk a0 = a :=
  Quot.exists_rep a


/-- A `MonoidHom` maps conjugacy classes of one group to conjugacy classes of another. -/
def map (f : α →* β) : ConjClasses α → ConjClasses β :=
  Quotient.lift (ConjClasses.mk ∘ f) fun _ _ ab => mk_eq_mk_iff_isConj.2 (f.map_isConj ab)


theorem map_surjective {f : α →* β} (hf : Function.Surjective f) :
    Function.Surjective (ConjClasses.map f) := by
  /-
    α : Type u
    β : Type v
    inst✝¹ : Monoid α
    inst✝ : Monoid β
    f : MonoidHom α β
    hf : Function.Surjective ⇑f
    ⊢ Function.Surjective (ConjClasses.map f)
  -/
  intro b
  /-
    α : Type u
    β : Type v
    inst✝¹ : Monoid α
    inst✝ : Monoid β
    f : MonoidHom α β
    hf : Function.Surjective ⇑f
    b : ConjClasses β
    ⊢ Exists fun a => Eq (ConjClasses.map f a) b
  -/
  obtain ⟨b, rfl⟩ := ConjClasses.mk_surjective b
  /-
    case intro
    α : Type u
    β : Type v
    inst✝¹ : Monoid α
    inst✝ : Monoid β
    f : MonoidHom α β
    hf : Function.Surjective ⇑f
    b : β
    ⊢ Exists fun a => Eq (ConjClasses.map f a) (ConjClasses.mk b)
  -/
  obtain ⟨a, rfl⟩ := hf b
  /-
    case intro.intro
    α : Type u
    β : Type v
    inst✝¹ : Monoid α
    inst✝ : Monoid β
    f : MonoidHom α β
    hf : Function.Surjective ⇑f
    a : α
    ⊢ Exists fun a_1 => Eq (ConjClasses.map f a_1) (ConjClasses.mk (f a))
  -/
  exact ⟨ConjClasses.mk a, rfl⟩
  /-
    🎉 no goals
  -/

-- Porting note: This has not been adapted to mathlib4, is it still accurate?

instance (priority := 900) [DecidableRel (IsConj : α → α → Prop)] : DecidableEq (ConjClasses α) :=
  inferInstanceAs <| DecidableEq <| Quotient (IsConj.setoid α)


theorem mk_injective : Function.Injective (@ConjClasses.mk α _) := fun _ _ =>
  (mk_eq_mk_iff_isConj.trans isConj_iff_eq).1


theorem mk_bijective : Function.Bijective (@ConjClasses.mk α _) :=
  ⟨mk_injective, mk_surjective⟩


/-- The bijection between a `CommGroup` and its `ConjClasses`. -/
def mkEquiv : α ≃ ConjClasses α :=
  ⟨ConjClasses.mk, Quotient.lift id fun (_ : α) _ => isConj_iff_eq.1, Quotient.lift_mk _ _, by
    /-
      α : Type u
      β : Type v
      inst✝ : CommMonoid α
      ⊢ Function.RightInverse (Quotient.lift id ⋯) ConjClasses.mk
    -/
    rw [Function.RightInverse, Function.LeftInverse, forall_isConj]
    /-
      α : Type u
      β : Type v
      inst✝ : CommMonoid α
      ⊢ ∀ (a : α), Eq (ConjClasses.mk (Quotient.lift id ⋯ (ConjClasses.mk a))) (Conj …
    -/
    intro x
    /-
      α : Type u
      β : Type v
      inst✝ : CommMonoid α
      x : α
      ⊢ Eq (ConjClasses.mk (Quotient.lift id ⋯ (ConjClasses.mk x))) (ConjClasses.mk x)
    -/
    rw [← quotient_mk_eq_mk, ← quotient_mk_eq_mk, Quotient.lift_mk, id]⟩
    /-
      🎉 no goals
    -/


/-- Given an element `a`, `conjugatesOf a` is the set of conjugates. -/
def conjugatesOf (a : α) : Set α :=
  { b | IsConj a b }


theorem mem_conjugatesOf_self {a : α} : a ∈ conjugatesOf a :=
  IsConj.refl _


theorem IsConj.conjugatesOf_eq {a b : α} (ab : IsConj a b) : conjugatesOf a = conjugatesOf b :=
  Set.ext fun _ => ⟨fun ag => ab.symm.trans ag, fun bg => ab.trans bg⟩


theorem isConj_iff_conjugatesOf_eq {a b : α} : IsConj a b ↔ conjugatesOf a = conjugatesOf b :=
  ⟨IsConj.conjugatesOf_eq, fun h => by
    /-
      α : Type u
      inst✝ : Monoid α
      a b : α
      h : Eq (conjugatesOf a) (conjugatesOf b)
      ⊢ IsConj a b
    -/
    have ha := @mem_conjugatesOf_self _ _ b -- Porting note: added `@`.
    /-
      α : Type u
      inst✝ : Monoid α
      a b : α
      h : Eq (conjugatesOf a) (conjugatesOf b)
      ha : Membership.mem (conjugatesOf b) b
      ⊢ IsConj a b
    -/
    rwa [← h] at ha⟩
    /-
      🎉 no goals
    -/


/-- Given a conjugacy class `a`, `carrier a` is the set it represents. -/
def carrier : ConjClasses α → Set α :=
  Quotient.lift conjugatesOf fun (_ : α) _ ab => IsConj.conjugatesOf_eq ab


theorem mem_carrier_mk {a : α} : a ∈ carrier (ConjClasses.mk a) :=
  IsConj.refl _


theorem mem_carrier_iff_mk_eq {a : α} {b : ConjClasses α} :
    a ∈ carrier b ↔ ConjClasses.mk a = b := by
  /-
    α : Type u
    inst✝ : Monoid α
    a : α
    b : ConjClasses α
    ⊢ Iff (Membership.mem b.carrier a) (Eq (ConjClasses.mk a) b)
  -/
  revert b
  /-
    α : Type u
    inst✝ : Monoid α
    a : α
    ⊢ ∀ {b : ConjClasses α}, Iff (Membership.mem b.carrier a) (Eq (ConjClasses.mk  …
  -/
  rw [forall_isConj]
  /-
    α : Type u
    inst✝ : Monoid α
    a : α
    ⊢ ∀ (a_1 : α), Iff (Membership.mem (ConjClasses.mk a_1).carrier a) (Eq (ConjCl …
  -/
  intro b
  /-
    α : Type u
    inst✝ : Monoid α
    a b : α
    ⊢ Iff (Membership.mem (ConjClasses.mk b).carrier a) (Eq (ConjClasses.mk a) (Co …
  -/
  rw [carrier, eq_comm, mk_eq_mk_iff_isConj, ← quotient_mk_eq_mk, Quotient.lift_mk]
  /-
    α : Type u
    inst✝ : Monoid α
    a b : α
    ⊢ Iff (Membership.mem (conjugatesOf b) a) (IsConj b a)
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem carrier_eq_preimage_mk {a : ConjClasses α} : a.carrier = ConjClasses.mk ⁻¹' {a} :=
  Set.ext fun _ => mem_carrier_iff_mk_eq


