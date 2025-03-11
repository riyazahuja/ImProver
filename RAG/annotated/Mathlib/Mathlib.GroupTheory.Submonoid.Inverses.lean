@[to_additive]
noncomputable instance [Monoid M] : Group (IsUnit.submonoid M) :=
  { inferInstanceAs (Monoid (IsUnit.submonoid M)) with
    inv := fun x ↦ ⟨x.prop.unit⁻¹.val, x.prop.unit⁻¹.isUnit⟩
    inv_mul_cancel := fun x ↦
      Subtype.ext ((Units.val_mul x.prop.unit⁻¹ _).trans x.prop.unit.inv_val) }


@[to_additive]
noncomputable instance [CommMonoid M] : CommGroup (IsUnit.submonoid M) :=
  { inferInstanceAs (Group (IsUnit.submonoid M)) with
                             /-
                               M : Type u_1
                               inst✝ : CommMonoid M
                               a b : Subtype fun x => Membership.mem (IsUnit.submonoid M) x
                               ⊢ Eq (HMul.hMul a b) (HMul.hMul b a)
                             -/
    mul_comm := fun a b ↦ by convert mul_comm a b }
                             /-
                               🎉 no goals
                             -/


@[to_additive]
theorem IsUnit.Submonoid.coe_inv [Monoid M] (x : IsUnit.submonoid M) :
    ↑x⁻¹ = (↑x.prop.unit⁻¹ : M) :=
  rfl


/-- `S.leftInv` is the submonoid containing all the left inverses of `S`. -/
@[to_additive
      "`S.leftNeg` is the additive submonoid containing all the left additive inverses of `S`."]
def leftInv : Submonoid M where
  carrier := { x : M | ∃ y : S, x * y = 1 }
  one_mem' := ⟨1, mul_one 1⟩
  mul_mem' := fun {a} _b ⟨a', ha⟩ ⟨b', hb⟩ ↦
                 /-
                   M : Type u_1
                   inst✝ : Monoid M
                   S : Submonoid M
                   a _b : M
                   x✝¹ : Membership.mem (setOf fun x => Exists fun y => Eq (HMul.hMul x ↑y) 1) a
                   x✝ : Membership.mem (setOf fun x => Exists fun y => Eq (HMul.hMul x ↑y) 1) _b
                   a' : Subtype fun x => Membership.mem S x
                   ha : Eq (HMul.hMul a ↑a') 1
                   b' : Subtype fun x => Membership.mem S x
                   hb : Eq (HMul.hMul _b ↑b') 1
                   ⊢ Eq (HMul.hMul (HMul.hMul a _b) ↑(HMul.hMul b' a')) 1
                 -/
    ⟨b' * a', by simp only [coe_mul, ← mul_assoc, mul_assoc a, hb, mul_one, ha]⟩
                 /-
                   🎉 no goals
                 -/


@[to_additive]
theorem leftInv_leftInv_le : S.leftInv.leftInv ≤ S := by
  /-
    M : Type u_1
    inst✝ : Monoid M
    S : Submonoid M
    ⊢ LE.le S.leftInv.leftInv S
  -/
  rintro x ⟨⟨y, z, h₁⟩, h₂ : x * y = 1⟩
  /-
    case intro.mk.intro
    M : Type u_1
    inst✝ : Monoid M
    S : Submonoid M
    x y : M
    z : Subtype fun x => Membership.mem S x
    h₁ : Eq (HMul.hMul y ↑z) 1
    h₂ : Eq (HMul.hMul x y) 1
    ⊢ Membership.mem S x
  -/
  convert z.prop
  /-
    case h.e'_5
    M : Type u_1
    inst✝ : Monoid M
    S : Submonoid M
    x y : M
    z : Subtype fun x => Membership.mem S x
    h₁ : Eq (HMul.hMul y ↑z) 1
    h₂ : Eq (HMul.hMul x y) 1
    ⊢ Eq x ↑z
  -/
  rw [← mul_one x, ← h₁, ← mul_assoc, h₂, one_mul]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem unit_mem_leftInv (x : Mˣ) (hx : (x : M) ∈ S) : ((x⁻¹ : _) : M) ∈ S.leftInv :=
  ⟨⟨x, hx⟩, x.inv_val⟩


@[to_additive]
theorem leftInv_leftInv_eq (hS : S ≤ IsUnit.submonoid M) : S.leftInv.leftInv = S := by
  /-
    M : Type u_1
    inst✝ : Monoid M
    S : Submonoid M
    hS : LE.le S (IsUnit.submonoid M)
    ⊢ Eq S.leftInv.leftInv S
  -/
  refine le_antisymm S.leftInv_leftInv_le ?_
  /-
    M : Type u_1
    inst✝ : Monoid M
    S : Submonoid M
    hS : LE.le S (IsUnit.submonoid M)
    ⊢ LE.le S S.leftInv.leftInv
  -/
  intro x hx
  have : x = ((hS hx).unit⁻¹⁻¹ : Mˣ) := by
    rw [inv_inv (hS hx).unit]
    rfl
  /-
    M : Type u_1
    inst✝ : Monoid M
    S : Submonoid M
    hS : LE.le S (IsUnit.submonoid M)
    x : M
    hx : Membership.mem S x
    this : Eq x ↑(Inv.inv (Inv.inv (IsUnit.unit ⋯)))
    ⊢ Membership.mem S.leftInv.leftInv x
  -/
  rw [this]
  /-
    M : Type u_1
    inst✝ : Monoid M
    S : Submonoid M
    hS : LE.le S (IsUnit.submonoid M)
    x : M
    hx : Membership.mem S x
    this : Eq x ↑(Inv.inv (Inv.inv (IsUnit.unit ⋯)))
    ⊢ Membership.mem S.leftInv.leftInv ↑(Inv.inv (Inv.inv (IsUnit.unit ⋯)))
  -/
  exact S.leftInv.unit_mem_leftInv _ (S.unit_mem_leftInv _ hx)
  /-
    🎉 no goals
  -/


/-- The function from `S.leftInv` to `S` sending an element to its right inverse in `S`.
This is a `MonoidHom` when `M` is commutative. -/
@[to_additive
      "The function from `S.leftAdd` to `S` sending an element to its right additive
inverse in `S`. This is an `AddMonoidHom` when `M` is commutative."]
noncomputable def fromLeftInv : S.leftInv → S := fun x ↦ x.prop.choose


@[to_additive (attr := simp)]
theorem mul_fromLeftInv (x : S.leftInv) : (x : M) * S.fromLeftInv x = 1 :=
  x.prop.choose_spec


@[to_additive (attr := simp)]
theorem fromLeftInv_one : S.fromLeftInv 1 = 1 :=
  (one_mul _).symm.trans (Subtype.eq <| S.mul_fromLeftInv 1)


@[to_additive (attr := simp)]
theorem fromLeftInv_mul (x : S.leftInv) : (S.fromLeftInv x : M) * x = 1 := by
  /-
    M : Type u_1
    inst✝ : CommMonoid M
    S : Submonoid M
    x : Subtype fun x => Membership.mem S.leftInv x
    ⊢ Eq (HMul.hMul ↑(S.fromLeftInv x) ↑x) 1
  -/
  rw [mul_comm, mul_fromLeftInv]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem leftInv_le_isUnit : S.leftInv ≤ IsUnit.submonoid M := fun x ⟨y, hx⟩ ↦
  ⟨⟨x, y, hx, mul_comm x y ▸ hx⟩, rfl⟩


@[to_additive]
theorem fromLeftInv_eq_iff (a : S.leftInv) (b : M) :
    (S.fromLeftInv a : M) = b ↔ (a : M) * b = 1 := by
  /-
    M : Type u_1
    inst✝ : CommMonoid M
    S : Submonoid M
    a : Subtype fun x => Membership.mem S.leftInv x
    b : M
    ⊢ Iff (Eq (↑(S.fromLeftInv a)) b) (Eq (HMul.hMul (↑a) b) 1)
  -/
  rw [← IsUnit.mul_right_inj (leftInv_le_isUnit _ a.prop), S.mul_fromLeftInv, eq_comm]
  /-
    🎉 no goals
  -/


/-- The `MonoidHom` from `S.leftInv` to `S` sending an element to its right inverse in `S`. -/
@[to_additive (attr := simps)
    "The `AddMonoidHom` from `S.leftNeg` to `S` sending an element to its
    right additive inverse in `S`."]
noncomputable def fromCommLeftInv : S.leftInv →* S where
  toFun := S.fromLeftInv
  map_one' := S.fromLeftInv_one
  map_mul' x y :=
    Subtype.ext <| by
      rw [fromLeftInv_eq_iff, mul_comm x, Submonoid.coe_mul, Submonoid.coe_mul, mul_assoc, ←
        mul_assoc (x : M), mul_fromLeftInv, one_mul, mul_fromLeftInv]


/-- The submonoid of pointwise inverse of `S` is `MulEquiv` to `S`. -/
@[to_additive (attr := simps apply) "The additive submonoid of pointwise additive inverse of `S` is
`AddEquiv` to `S`."]
noncomputable def leftInvEquiv : S.leftInv ≃* S :=
  { S.fromCommLeftInv with
                                               /-
                                                 M : Type u_1
                                                 inst✝ : CommMonoid M
                                                 S : Submonoid M
                                                 hS : LE.le S (IsUnit.submonoid M)
                                                 x : Subtype fun x => Membership.mem S x
                                                 ⊢ Eq (HMul.hMul ↑(Inv.inv (IsUnit.unit ⋯)) ↑x) 1
                                               -/
    invFun := fun x ↦ ⟨↑(hS x.2).unit⁻¹, x, by simp⟩
                                               /-
                                                 🎉 no goals
                                               -/
    left_inv := by
      /-
        M : Type u_1
        inst✝ : CommMonoid M
        S : Submonoid M
        hS : LE.le S (IsUnit.submonoid M)
        ⊢ Function.LeftInverse (fun x => ⟨↑(Inv.inv (IsUnit.unit ⋯)), ⋯⟩) (↑__src✝).to …
      -/
      intro x
      /-
        M : Type u_1
        inst✝ : CommMonoid M
        S : Submonoid M
        hS : LE.le S (IsUnit.submonoid M)
        x : Subtype fun x => Membership.mem S.leftInv x
        ⊢ Eq ((fun x => ⟨↑(Inv.inv (IsUnit.unit ⋯)), ⋯⟩) ((↑__src✝).toFun x)) x
      -/
      ext
      /-
        case a
        M : Type u_1
        inst✝ : CommMonoid M
        S : Submonoid M
        hS : LE.le S (IsUnit.submonoid M)
        x : Subtype fun x => Membership.mem S.leftInv x
        ⊢ Eq ↑((fun x => ⟨↑(Inv.inv (IsUnit.unit ⋯)), ⋯⟩) ((↑__src✝).toFun x)) ↑x
      -/
      simp [← Units.mul_eq_one_iff_inv_eq]
      /-
        🎉 no goals
      -/
    right_inv := by
      /-
        M : Type u_1
        inst✝ : CommMonoid M
        S : Submonoid M
        hS : LE.le S (IsUnit.submonoid M)
        ⊢ Function.RightInverse (fun x => ⟨↑(Inv.inv (IsUnit.unit ⋯)), ⋯⟩) (↑__src✝).t …
      -/
      rintro ⟨x, hx⟩
      /-
        case mk
        M : Type u_1
        inst✝ : CommMonoid M
        S : Submonoid M
        hS : LE.le S (IsUnit.submonoid M)
        x : M
        hx : Membership.mem S x
        ⊢ Eq ((↑__src✝).toFun ((fun x => ⟨↑(Inv.inv (IsUnit.unit ⋯)), ⋯⟩) ⟨x, hx⟩)) ⟨x …
      -/
      ext
      /-
        case mk.a
        M : Type u_1
        inst✝ : CommMonoid M
        S : Submonoid M
        hS : LE.le S (IsUnit.submonoid M)
        x : M
        hx : Membership.mem S x
        ⊢ Eq ↑((↑__src✝).toFun ((fun x => ⟨↑(Inv.inv (IsUnit.unit ⋯)), ⋯⟩) ⟨x, hx⟩)) ↑ …
      -/
      simp [fromLeftInv_eq_iff] }
      /-
        🎉 no goals
      -/


@[to_additive (attr := simp)]
theorem fromLeftInv_leftInvEquiv_symm (x : S) : S.fromLeftInv ((S.leftInvEquiv hS).symm x) = x :=
  (S.leftInvEquiv hS).right_inv x


@[to_additive (attr := simp)]
theorem leftInvEquiv_symm_fromLeftInv (x : S.leftInv) :
    (S.leftInvEquiv hS).symm (S.fromLeftInv x) = x :=
  (S.leftInvEquiv hS).left_inv x


@[to_additive]
theorem leftInvEquiv_mul (x : S.leftInv) : (S.leftInvEquiv hS x : M) * x = 1 := by
  /-
    M : Type u_1
    inst✝ : CommMonoid M
    S : Submonoid M
    hS : LE.le S (IsUnit.submonoid M)
    x : Subtype fun x => Membership.mem S.leftInv x
    ⊢ Eq (HMul.hMul ↑((S.leftInvEquiv hS) x) ↑x) 1
  -/
  simpa only [leftInvEquiv_apply, fromCommLeftInv] using fromLeftInv_mul S x
  /-
    🎉 no goals
  -/


@[to_additive]
theorem mul_leftInvEquiv (x : S.leftInv) : (x : M) * S.leftInvEquiv hS x = 1 := by
  /-
    M : Type u_1
    inst✝ : CommMonoid M
    S : Submonoid M
    hS : LE.le S (IsUnit.submonoid M)
    x : Subtype fun x => Membership.mem S.leftInv x
    ⊢ Eq (HMul.hMul ↑x ↑((S.leftInvEquiv hS) x)) 1
  -/
  simp only [leftInvEquiv_apply, fromCommLeftInv, mul_fromLeftInv]
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)]
theorem leftInvEquiv_symm_mul (x : S) : ((S.leftInvEquiv hS).symm x : M) * x = 1 := by
  /-
    M : Type u_1
    inst✝ : CommMonoid M
    S : Submonoid M
    hS : LE.le S (IsUnit.submonoid M)
    x : Subtype fun x => Membership.mem S x
    ⊢ Eq (HMul.hMul ↑((S.leftInvEquiv hS).symm x) ↑x) 1
  -/
  convert S.mul_leftInvEquiv hS ((S.leftInvEquiv hS).symm x)
  /-
    case h.e'_2.h.e'_6.h.e'_3
    M : Type u_1
    inst✝ : CommMonoid M
    S : Submonoid M
    hS : LE.le S (IsUnit.submonoid M)
    x : Subtype fun x => Membership.mem S x
    ⊢ Eq x ((S.leftInvEquiv hS) ((S.leftInvEquiv hS).symm x))
  -/
  simp
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)]
theorem mul_leftInvEquiv_symm (x : S) : (x : M) * (S.leftInvEquiv hS).symm x = 1 := by
  /-
    M : Type u_1
    inst✝ : CommMonoid M
    S : Submonoid M
    hS : LE.le S (IsUnit.submonoid M)
    x : Subtype fun x => Membership.mem S x
    ⊢ Eq (HMul.hMul ↑x ↑((S.leftInvEquiv hS).symm x)) 1
  -/
  convert S.leftInvEquiv_mul hS ((S.leftInvEquiv hS).symm x)
  /-
    case h.e'_2.h.e'_5.h.e'_3
    M : Type u_1
    inst✝ : CommMonoid M
    S : Submonoid M
    hS : LE.le S (IsUnit.submonoid M)
    x : Subtype fun x => Membership.mem S x
    ⊢ Eq x ((S.leftInvEquiv hS) ((S.leftInvEquiv hS).symm x))
  -/
  simp
  /-
    🎉 no goals
  -/


@[to_additive]
theorem leftInv_eq_inv : S.leftInv = S⁻¹ :=
  Submonoid.ext fun _ ↦
    ⟨fun h ↦ Submonoid.mem_inv.mpr ((inv_eq_of_mul_eq_one_right h.choose_spec).symm ▸
      h.choose.prop),
      fun h ↦ ⟨⟨_, h⟩, mul_inv_cancel _⟩⟩


@[to_additive (attr := simp)]
theorem fromLeftInv_eq_inv (x : S.leftInv) : (S.fromLeftInv x : M) = (x : M)⁻¹ := by
  /-
    M : Type u_1
    inst✝ : Group M
    S : Submonoid M
    x : Subtype fun x => Membership.mem S.leftInv x
    ⊢ Eq (↑(S.fromLeftInv x)) (Inv.inv ↑x)
  -/
  rw [← mul_right_inj (x : M), mul_inv_cancel, mul_fromLeftInv]
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)]
theorem leftInvEquiv_symm_eq_inv (x : S) : ((S.leftInvEquiv hS).symm x : M) = (x : M)⁻¹ := by
  /-
    M : Type u_1
    inst✝ : CommGroup M
    S : Submonoid M
    hS : LE.le S (IsUnit.submonoid M)
    x : Subtype fun x => Membership.mem S x
    ⊢ Eq (↑((S.leftInvEquiv hS).symm x)) (Inv.inv ↑x)
  -/
  rw [← mul_right_inj (x : M), mul_inv_cancel, mul_leftInvEquiv_symm]
  /-
    🎉 no goals
  -/


