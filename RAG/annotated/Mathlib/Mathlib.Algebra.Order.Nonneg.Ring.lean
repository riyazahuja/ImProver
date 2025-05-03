instance orderedAddCommMonoid [OrderedAddCommMonoid α] : OrderedAddCommMonoid { x : α // 0 ≤ x } :=
  Subtype.coe_injective.orderedAddCommMonoid _ Nonneg.coe_zero (fun _ _ => rfl) fun _ _ => rfl


instance linearOrderedAddCommMonoid [LinearOrderedAddCommMonoid α] :
    LinearOrderedAddCommMonoid { x : α // 0 ≤ x } :=
  Subtype.coe_injective.linearOrderedAddCommMonoid _ Nonneg.coe_zero
    (fun _ _ => rfl) (fun _ _ => rfl)
    (fun _ _ => rfl) fun _ _ => rfl


instance orderedCancelAddCommMonoid [OrderedCancelAddCommMonoid α] :
    OrderedCancelAddCommMonoid { x : α // 0 ≤ x } :=
  Subtype.coe_injective.orderedCancelAddCommMonoid _ Nonneg.coe_zero (fun _ _ => rfl) fun _ _ => rfl


instance linearOrderedCancelAddCommMonoid [LinearOrderedCancelAddCommMonoid α] :
    LinearOrderedCancelAddCommMonoid { x : α // 0 ≤ x } :=
  Subtype.coe_injective.linearOrderedCancelAddCommMonoid _ Nonneg.coe_zero
    (fun _ _ => rfl) (fun _ _ => rfl)
    (fun _ _ => rfl) fun _ _ => rfl


instance orderedSemiring [OrderedSemiring α] : OrderedSemiring { x : α // 0 ≤ x } :=
  Subtype.coe_injective.orderedSemiring _ Nonneg.coe_zero Nonneg.coe_one
    (fun _ _ => rfl) (fun _ _=> rfl) (fun _ _ => rfl)
    (fun _ _ => rfl) fun _ => rfl


instance strictOrderedSemiring [StrictOrderedSemiring α] :
    StrictOrderedSemiring { x : α // 0 ≤ x } :=
  Subtype.coe_injective.strictOrderedSemiring _ Nonneg.coe_zero Nonneg.coe_one
    (fun _ _ => rfl) (fun _ _ => rfl)
    (fun _ _ => rfl) (fun _ _ => rfl) fun _ => rfl


instance orderedCommSemiring [OrderedCommSemiring α] : OrderedCommSemiring { x : α // 0 ≤ x } :=
  Subtype.coe_injective.orderedCommSemiring _ Nonneg.coe_zero Nonneg.coe_one
    (fun _ _ => rfl) (fun _ _ => rfl)
    (fun _ _ => rfl) (fun _ _ => rfl) fun _ => rfl


instance orderedCommMonoid [OrderedCommSemiring α] : OrderedCommMonoid { x : α // 0 ≤ x } where
  mul_le_mul_left a _ h c := mul_le_mul le_rfl h a.prop c.prop


instance strictOrderedCommSemiring [StrictOrderedCommSemiring α] :
    StrictOrderedCommSemiring { x : α // 0 ≤ x } :=
  Subtype.coe_injective.strictOrderedCommSemiring _ Nonneg.coe_zero Nonneg.coe_one
    (fun _ _ => rfl) (fun _ _ => rfl)
    (fun _ _ => rfl) (fun _ _ => rfl) fun _ => rfl


instance existsAddOfLE [StrictOrderedCommSemiring α] [ExistsAddOfLE α] :
    ExistsAddOfLE { x : α // 0 ≤ x } :=
  ⟨fun {a b} h ↦ by
    /-
      α : Type u_1
      inst✝¹ : StrictOrderedCommSemiring α
      inst✝ : ExistsAddOfLE α
      a b : Subtype fun x => LE.le 0 x
      h : LE.le a b
      ⊢ Exists fun c => Eq b (HAdd.hAdd a c)
    -/
    rw [← Subtype.coe_le_coe] at h
    /-
      α : Type u_1
      inst✝¹ : StrictOrderedCommSemiring α
      inst✝ : ExistsAddOfLE α
      a b : Subtype fun x => LE.le 0 x
      h : LE.le ↑a ↑b
      ⊢ Exists fun c => Eq b (HAdd.hAdd a c)
    -/
    obtain ⟨c, hc⟩ := exists_add_of_le h
    /-
      case intro
      α : Type u_1
      inst✝¹ : StrictOrderedCommSemiring α
      inst✝ : ExistsAddOfLE α
      a b : Subtype fun x => LE.le 0 x
      h : LE.le ↑a ↑b
      c : α
      hc : Eq (↑b) (HAdd.hAdd (↑a) c)
      ⊢ Exists fun c => Eq b (HAdd.hAdd a c)
    -/
    refine ⟨⟨c, ?_⟩, by simp [Subtype.ext_iff, hc]⟩
    /-
      case intro
      α : Type u_1
      inst✝¹ : StrictOrderedCommSemiring α
      inst✝ : ExistsAddOfLE α
      a b : Subtype fun x => LE.le 0 x
      h : LE.le ↑a ↑b
      c : α
      hc : Eq (↑b) (HAdd.hAdd (↑a) c)
      ⊢ LE.le 0 c
    -/
    rw [← add_zero a.val, hc] at h
    /-
      case intro
      α : Type u_1
      inst✝¹ : StrictOrderedCommSemiring α
      inst✝ : ExistsAddOfLE α
      a b : Subtype fun x => LE.le 0 x
      c : α
      h : LE.le (HAdd.hAdd (↑a) 0) (HAdd.hAdd (↑a) c)
      hc : Eq (↑b) (HAdd.hAdd (↑a) c)
      ⊢ LE.le 0 c
    -/
    exact le_of_add_le_add_left h⟩
    /-
      🎉 no goals
    -/


instance nontrivial [LinearOrderedSemiring α] : Nontrivial { x : α // 0 ≤ x } :=
  ⟨⟨0, 1, fun h => zero_ne_one (congr_arg Subtype.val h)⟩⟩


instance linearOrderedSemiring [LinearOrderedSemiring α] :
    LinearOrderedSemiring { x : α // 0 ≤ x } :=
  Subtype.coe_injective.linearOrderedSemiring _ Nonneg.coe_zero Nonneg.coe_one
    (fun _ _ => rfl) (fun _ _ => rfl)
    (fun _ _ => rfl) (fun _ _ => rfl) (fun _ => rfl) (fun _ _ => rfl) fun _ _ => rfl


instance linearOrderedCommMonoidWithZero [LinearOrderedCommSemiring α] :
    LinearOrderedCommMonoidWithZero { x : α // 0 ≤ x } :=
  { Nonneg.linearOrderedSemiring, Nonneg.orderedCommSemiring with
    mul_le_mul_left := fun _ _ h c ↦ mul_le_mul_of_nonneg_left h c.prop }


instance canonicallyOrderedAddCommMonoid [OrderedRing α] :
    CanonicallyOrderedAddCommMonoid { x : α // 0 ≤ x } :=
  { Nonneg.orderedAddCommMonoid, Nonneg.orderBot with
    le_self_add := fun _ b => le_add_of_nonneg_right b.2
    exists_add_of_le := fun {a b} h =>
      ⟨⟨b - a, sub_nonneg_of_le h⟩, Subtype.ext (add_sub_cancel _ _).symm⟩ }


instance canonicallyOrderedCommSemiring [OrderedCommRing α] [NoZeroDivisors α] :
    CanonicallyOrderedCommSemiring { x : α // 0 ≤ x } :=
  { Nonneg.canonicallyOrderedAddCommMonoid, Nonneg.orderedCommSemiring with
    eq_zero_or_eq_zero_of_mul_eq_zero := by
      /-
        α : Type u_1
        inst✝¹ : OrderedCommRing α
        inst✝ : NoZeroDivisors α
        ⊢ ∀ {a b : Subtype fun x => LE.le 0 x}, Eq (HMul.hMul a b) 0 → Or (Eq a 0) (Eq …
      -/
      rintro ⟨a, ha⟩ ⟨b, hb⟩
      /-
        case mk.mk
        α : Type u_1
        inst✝¹ : OrderedCommRing α
        inst✝ : NoZeroDivisors α
        a : α
        ha : LE.le 0 a
        b : α
        hb : LE.le 0 b
        ⊢ Eq (HMul.hMul ⟨a, ha⟩ ⟨b, hb⟩) 0 → Or (Eq ⟨a, ha⟩ 0) (Eq ⟨b, hb⟩ 0)
      -/
      simp only [mk_mul_mk, mk_eq_zero, mul_eq_zero, imp_self]}
      /-
        🎉 no goals
      -/


instance canonicallyLinearOrderedAddCommMonoid [LinearOrderedRing α] :
    CanonicallyLinearOrderedAddCommMonoid { x : α // 0 ≤ x } :=
  { Subtype.instLinearOrder _, Nonneg.canonicallyOrderedAddCommMonoid with }


instance orderedSub [LinearOrderedRing α] : OrderedSub { x : α // 0 ≤ x } :=
  ⟨by
    /-
      α : Type u_1
      inst✝ : LinearOrderedRing α
      ⊢ ∀ (a b c : Subtype fun x => LE.le 0 x), Iff (LE.le (HSub.hSub a b) c) (LE.le …
    -/
    rintro ⟨a, ha⟩ ⟨b, hb⟩ ⟨c, hc⟩
    simp only [sub_le_iff_le_add, Subtype.mk_le_mk, mk_sub_mk, mk_add_mk, toNonneg_le,
      Subtype.coe_mk]⟩


