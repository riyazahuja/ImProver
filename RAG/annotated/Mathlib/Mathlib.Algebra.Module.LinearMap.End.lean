/-- Linear endomorphisms of a module, with associated ring structure
`Module.End.semiring` and algebra structure `Module.End.algebra`. -/
abbrev Module.End (R : Type u) (M : Type v) [Semiring R] [AddCommMonoid M] [Module R M] :=
  M →ₗ[R] M


instance : One (Module.End R M) := ⟨LinearMap.id⟩


instance : Mul (Module.End R M) := ⟨fun f g => LinearMap.comp f g⟩


theorem one_eq_id : (1 : Module.End R M) = id := rfl


theorem mul_eq_comp (f g : Module.End R M) : f * g = f.comp g := rfl


@[simp]
theorem one_apply (x : M) : (1 : Module.End R M) x = x := rfl


@[simp]
theorem mul_apply (f g : Module.End R M) (x : M) : (f * g) x = f (g x) := rfl


theorem coe_one : ⇑(1 : Module.End R M) = _root_.id := rfl


theorem coe_mul (f g : Module.End R M) : ⇑(f * g) = f ∘ g := rfl


instance _root_.Module.End.instNontrivial [Nontrivial M] : Nontrivial (Module.End R M) := by
  /-
    R : Type u_1
    R₂ : Type u_2
    S : Type u_3
    M : Type u_4
    M₁ : Type u_5
    M₂ : Type u_6
    M₃ : Type u_7
    N₁ : Type u_8
    inst✝⁵ : Semiring R
    inst✝⁴ : AddCommMonoid M
    inst✝³ : AddCommGroup N₁
    inst✝² : Module R M
    inst✝¹ : Module R N₁
    inst✝ : Nontrivial M
    ⊢ Nontrivial (Module.End R M)
  -/
  obtain ⟨m, ne⟩ := exists_ne (0 : M)
  /-
    case intro
    R : Type u_1
    R₂ : Type u_2
    S : Type u_3
    M : Type u_4
    M₁ : Type u_5
    M₂ : Type u_6
    M₃ : Type u_7
    N₁ : Type u_8
    inst✝⁵ : Semiring R
    inst✝⁴ : AddCommMonoid M
    inst✝³ : AddCommGroup N₁
    inst✝² : Module R M
    inst✝¹ : Module R N₁
    inst✝ : Nontrivial M
    m : M
    ne : Ne m 0
    ⊢ Nontrivial (Module.End R M)
  -/
  exact nontrivial_of_ne 1 0 fun p => ne (LinearMap.congr_fun p m)
  /-
    🎉 no goals
  -/


instance _root_.Module.End.monoid : Monoid (Module.End R M) where
  mul := (· * ·)
  one := (1 : M →ₗ[R] M)
  mul_assoc _ _ _ := LinearMap.ext fun _ ↦ rfl
  mul_one := comp_id
  one_mul := id_comp


instance _root_.Module.End.semiring : Semiring (Module.End R M) :=
  { AddMonoidWithOne.unary, Module.End.monoid, LinearMap.addCommMonoid with
    mul_zero := comp_zero
    zero_mul := zero_comp
    left_distrib := fun _ _ _ ↦ comp_add _ _ _
    right_distrib := fun _ _ _ ↦ add_comp _ _ _
    natCast := fun n ↦ n • (1 : M →ₗ[R] M)
    natCast_zero := zero_smul ℕ (1 : M →ₗ[R] M)
    natCast_succ := fun n ↦ AddMonoid.nsmul_succ n (1 : M →ₗ[R] M) }


/-- See also `Module.End.natCast_def`. -/
@[simp]
theorem _root_.Module.End.natCast_apply (n : ℕ) (m : M) : (↑n : Module.End R M) m = n • m := rfl


@[simp]
theorem _root_.Module.End.ofNat_apply (n : ℕ) [n.AtLeastTwo] (m : M) :
    (no_index (OfNat.ofNat n) : Module.End R M) m = OfNat.ofNat n • m := rfl


instance _root_.Module.End.ring : Ring (Module.End R N₁) :=
  { Module.End.semiring, LinearMap.addCommGroup with
    intCast := fun z ↦ z • (1 : N₁ →ₗ[R] N₁)
    intCast_ofNat := natCast_zsmul _
    intCast_negSucc := negSucc_zsmul _ }


/-- See also `Module.End.intCast_def`. -/
@[simp]
theorem _root_.Module.End.intCast_apply (z : ℤ) (m : N₁) : (z : Module.End R N₁) m = z • m :=
  rfl


instance _root_.Module.End.isScalarTower :
    IsScalarTower S (Module.End R M) (Module.End R M) :=
  ⟨smul_comp⟩


instance _root_.Module.End.smulCommClass [SMul S R] [IsScalarTower S R M] :
    SMulCommClass S (Module.End R M) (Module.End R M) :=
  ⟨fun s _ _ ↦ (comp_smul _ s _).symm⟩


instance _root_.Module.End.smulCommClass' [SMul S R] [IsScalarTower S R M] :
    SMulCommClass (Module.End R M) S (Module.End R M) :=
  SMulCommClass.symm _ _ _


theorem _root_.Module.End_isUnit_apply_inv_apply_of_isUnit
    {f : Module.End R M} (h : IsUnit f) (x : M) :
    f (h.unit.inv x) = x :=
                                 /-
                                   R : Type u_1
                                   M : Type u_4
                                   inst✝² : Semiring R
                                   inst✝¹ : AddCommMonoid M
                                   inst✝ : Module R M
                                   f : Module.End R M
                                   h : IsUnit f
                                   x : M
                                   ⊢ Eq ((HMul.hMul f h.unit.inv) x) x
                                 -/
  show (f * h.unit.inv) x = x by simp
                                 /-
                                   🎉 no goals
                                 -/


theorem _root_.Module.End_isUnit_inv_apply_apply_of_isUnit
    {f : Module.End R M} (h : IsUnit f) (x : M) :
    h.unit.inv (f x) = x :=
      /-
        R : Type u_1
        M : Type u_4
        inst✝² : Semiring R
        inst✝¹ : AddCommMonoid M
        inst✝ : Module R M
        f : Module.End R M
        h : IsUnit f
        x : M
        ⊢ Eq ((HMul.hMul h.unit.inv f) x) x
      -/
  (by simp : (h.unit.inv * f) x = x)
      /-
        🎉 no goals
      -/


theorem coe_pow (f : M →ₗ[R] M) (n : ℕ) : ⇑(f ^ n) = f^[n] := hom_coe_pow _ rfl (fun _ _ ↦ rfl) _ _


theorem pow_apply (f : M →ₗ[R] M) (n : ℕ) (m : M) : (f ^ n) m = f^[n] m := congr_fun (coe_pow f n) m


theorem pow_map_zero_of_le {f : Module.End R M} {m : M} {k l : ℕ} (hk : k ≤ l)
    (hm : (f ^ k) m = 0) : (f ^ l) m = 0 := by
  /-
    R : Type u_1
    M : Type u_4
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    f : Module.End R M
    m : M
    k l : Nat
    hk : LE.le k l
    hm : Eq ((HPow.hPow f k) m) 0
    ⊢ Eq ((HPow.hPow f l) m) 0
  -/
  rw [← Nat.sub_add_cancel hk, pow_add, mul_apply, hm, map_zero]
  /-
    🎉 no goals
  -/


theorem commute_pow_left_of_commute
    [Semiring R₂] [AddCommMonoid M₂] [Module R₂ M₂] {σ₁₂ : R →+* R₂}
    {f : M →ₛₗ[σ₁₂] M₂} {g : Module.End R M} {g₂ : Module.End R₂ M₂}
    (h : g₂.comp f = f.comp g) (k : ℕ) : (g₂ ^ k).comp f = f.comp (g ^ k) := by
  induction k with
  | zero => simp only [pow_zero, one_eq_id, id_comp, comp_id]
  | succ k ih => rw [pow_succ', pow_succ', LinearMap.mul_eq_comp, LinearMap.comp_assoc, ih,
    ← LinearMap.comp_assoc, h, LinearMap.comp_assoc, LinearMap.mul_eq_comp]


@[simp]
theorem id_pow (n : ℕ) : (id : M →ₗ[R] M) ^ n = id :=
  one_pow n


                                                                     /-
                                                                       R : Type u_1
                                                                       M : Type u_4
                                                                       inst✝² : Semiring R
                                                                       inst✝¹ : AddCommMonoid M
                                                                       inst✝ : Module R M
                                                                       f' : LinearMap (RingHom.id R) M M
                                                                       n : Nat
                                                                       ⊢ Eq (HPow.hPow f' (HAdd.hAdd n 1)) ((HPow.hPow f' n).comp f')
                                                                     -/
theorem iterate_succ (n : ℕ) : f' ^ (n + 1) = comp (f' ^ n) f' := by rw [pow_succ, mul_eq_comp]
                                                                     /-
                                                                       🎉 no goals
                                                                     -/


theorem iterate_surjective (h : Surjective f') : ∀ n : ℕ, Surjective (f' ^ n)
  | 0 => surjective_id
  | n + 1 => by
    /-
      R : Type u_1
      M : Type u_4
      inst✝² : Semiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      f' : LinearMap (RingHom.id R) M M
      h : Function.Surjective ⇑f'
      n : Nat
      ⊢ Function.Surjective ⇑(HPow.hPow f' (HAdd.hAdd n 1))
    -/
    rw [iterate_succ]
    /-
      R : Type u_1
      M : Type u_4
      inst✝² : Semiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      f' : LinearMap (RingHom.id R) M M
      h : Function.Surjective ⇑f'
      n : Nat
      ⊢ Function.Surjective ⇑((HPow.hPow f' n).comp f')
    -/
    exact (iterate_surjective h n).comp h
    /-
      🎉 no goals
    -/


theorem iterate_injective (h : Injective f') : ∀ n : ℕ, Injective (f' ^ n)
  | 0 => injective_id
  | n + 1 => by
    /-
      R : Type u_1
      M : Type u_4
      inst✝² : Semiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      f' : LinearMap (RingHom.id R) M M
      h : Function.Injective ⇑f'
      n : Nat
      ⊢ Function.Injective ⇑(HPow.hPow f' (HAdd.hAdd n 1))
    -/
    rw [iterate_succ]
    /-
      R : Type u_1
      M : Type u_4
      inst✝² : Semiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      f' : LinearMap (RingHom.id R) M M
      h : Function.Injective ⇑f'
      n : Nat
      ⊢ Function.Injective ⇑((HPow.hPow f' n).comp f')
    -/
    exact (iterate_injective h n).comp h
    /-
      🎉 no goals
    -/


theorem iterate_bijective (h : Bijective f') : ∀ n : ℕ, Bijective (f' ^ n)
  | 0 => bijective_id
  | n + 1 => by
    /-
      R : Type u_1
      M : Type u_4
      inst✝² : Semiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      f' : LinearMap (RingHom.id R) M M
      h : Function.Bijective ⇑f'
      n : Nat
      ⊢ Function.Bijective ⇑(HPow.hPow f' (HAdd.hAdd n 1))
    -/
    rw [iterate_succ]
    /-
      R : Type u_1
      M : Type u_4
      inst✝² : Semiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      f' : LinearMap (RingHom.id R) M M
      h : Function.Bijective ⇑f'
      n : Nat
      ⊢ Function.Bijective ⇑((HPow.hPow f' n).comp f')
    -/
    exact (iterate_bijective h n).comp h
    /-
      🎉 no goals
    -/


theorem injective_of_iterate_injective {n : ℕ} (hn : n ≠ 0) (h : Injective (f' ^ n)) :
    Injective f' := by
  /-
    R : Type u_1
    M : Type u_4
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    f' : LinearMap (RingHom.id R) M M
    n : Nat
    hn : Ne n 0
    h : Function.Injective ⇑(HPow.hPow f' n)
    ⊢ Function.Injective ⇑f'
  -/
  rw [← Nat.succ_pred_eq_of_pos (show 0 < n by omega), iterate_succ, coe_comp] at h
  /-
    R : Type u_1
    M : Type u_4
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    f' : LinearMap (RingHom.id R) M M
    n : Nat
    hn : Ne n 0
    h : Function.Injective (Function.comp ⇑(HPow.hPow f' n.pred) ⇑f')
    ⊢ Function.Injective ⇑f'
  -/
  exact h.of_comp
  /-
    🎉 no goals
  -/


theorem surjective_of_iterate_surjective {n : ℕ} (hn : n ≠ 0) (h : Surjective (f' ^ n)) :
    Surjective f' := by
  /-
    R : Type u_1
    M : Type u_4
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    f' : LinearMap (RingHom.id R) M M
    n : Nat
    hn : Ne n 0
    h : Function.Surjective ⇑(HPow.hPow f' n)
    ⊢ Function.Surjective ⇑f'
  -/
  rw [← Nat.succ_pred_eq_of_pos (Nat.pos_iff_ne_zero.mpr hn), pow_succ', coe_mul] at h
  /-
    R : Type u_1
    M : Type u_4
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    f' : LinearMap (RingHom.id R) M M
    n : Nat
    hn : Ne n 0
    h : Function.Surjective (Function.comp ⇑f' ⇑(HPow.hPow f' n.pred))
    ⊢ Function.Surjective ⇑f'
  -/
  exact Surjective.of_comp h
  /-
    🎉 no goals
  -/


/-- The tautological action by `Module.End R M` (aka `M →ₗ[R] M`) on `M`.

This generalizes `Function.End.applyMulAction`. -/
instance applyModule : Module (Module.End R M) M where
  smul := (· <| ·)
  smul_zero := LinearMap.map_zero
  smul_add := LinearMap.map_add
  add_smul := LinearMap.add_apply
  zero_smul := (LinearMap.zero_apply : ∀ m, (0 : M →ₗ[R] M) m = 0)
  one_smul _ := rfl
  mul_smul _ _ _ := rfl


@[simp]
protected theorem smul_def (f : Module.End R M) (a : M) : f • a = f a :=
  rfl


/-- `LinearMap.applyModule` is faithful. -/
instance apply_faithfulSMul : FaithfulSMul (Module.End R M) M :=
  ⟨LinearMap.ext⟩


instance apply_smulCommClass [SMul S R] [SMul S M] [IsScalarTower S R M] :
    SMulCommClass S (Module.End R M) M where
  smul_comm r e m := (e.map_smul_of_tower r m).symm


instance apply_smulCommClass' [SMul S R] [SMul S M] [IsScalarTower S R M] :
    SMulCommClass (Module.End R M) S M :=
  SMulCommClass.symm _ _ _


instance apply_isScalarTower [Monoid S] [DistribMulAction S M] [SMulCommClass R S M] :
    IsScalarTower S (Module.End R M) M :=
  ⟨fun _ _ _ ↦ rfl⟩


/-- Each element of the monoid defines a linear map.

This is a stronger version of `DistribMulAction.toAddMonoidHom`. -/
@[simps]
def toLinearMap (s : S) : M →ₗ[R] M where
  toFun := HSMul.hSMul s
  map_add' := smul_add s
  map_smul' _ _ := smul_comm _ _ _


/-- Each element of the monoid defines a module endomorphism.

This is a stronger version of `DistribMulAction.toAddMonoidEnd`. -/
@[simps]
def toModuleEnd : S →* Module.End R M where
  toFun := toLinearMap R M
  map_one' := LinearMap.ext <| one_smul _
  map_mul' _ _ := LinearMap.ext <| mul_smul _ _


/-- Each element of the semiring defines a module endomorphism.

This is a stronger version of `DistribMulAction.toModuleEnd`. -/
@[simps]
def toModuleEnd : S →+* Module.End R M :=
  { DistribMulAction.toModuleEnd R M with
    toFun := DistribMulAction.toLinearMap R M
    map_zero' := LinearMap.ext <| zero_smul S
    map_add' := fun _ _ ↦ LinearMap.ext <| add_smul _ _ }


/-- The canonical (semi)ring isomorphism from `Rᵐᵒᵖ` to `Module.End R R` induced by the right
multiplication. -/
@[simps]
def moduleEndSelf : Rᵐᵒᵖ ≃+* Module.End R R :=
  { Module.toModuleEnd R R with
    toFun := DistribMulAction.toLinearMap R R
    invFun := fun f ↦ MulOpposite.op (f 1)
    left_inv := mul_one
    right_inv := fun _ ↦ LinearMap.ext_ring <| one_mul _ }


/-- The canonical (semi)ring isomorphism from `R` to `Module.End Rᵐᵒᵖ R` induced by the left
multiplication. -/
@[simps]
def moduleEndSelfOp : R ≃+* Module.End Rᵐᵒᵖ R :=
  { Module.toModuleEnd _ _ with
    toFun := DistribMulAction.toLinearMap _ _
    invFun := fun f ↦ f 1
    left_inv := mul_one
    right_inv := fun _ ↦ LinearMap.ext_ring_op <| mul_one _ }


theorem End.natCast_def (n : ℕ) [AddCommMonoid N₁] [Module R N₁] :
    (↑n : Module.End R N₁) = Module.toModuleEnd R N₁ n :=
  rfl


theorem End.intCast_def (z : ℤ) [AddCommGroup N₁] [Module R N₁] :
    (z : Module.End R N₁) = Module.toModuleEnd R N₁ z :=
  rfl


/-- When `f` is an `R`-linear map taking values in `S`, then `fun b ↦ f b • x` is an `R`-linear
map. -/
def smulRight (f : M₁ →ₗ[R] S) (x : M) : M₁ →ₗ[R] M where
  toFun b := f b • x
                     /-
                       R : Type u_1
                       R₂ : Type u_2
                       S : Type u_3
                       M : Type u_4
                       M₁ : Type u_5
                       M₂ : Type u_6
                       M₃ : Type u_7
                       N₁ : Type u_8
                       inst✝⁸ : Semiring R
                       inst✝⁷ : AddCommMonoid M
                       inst✝⁶ : AddCommMonoid M₁
                       inst✝⁵ : Module R M
                       inst✝⁴ : Module R M₁
                       inst✝³ : Semiring S
                       inst✝² : Module R S
                       inst✝¹ : Module S M
                       inst✝ : IsScalarTower R S M
                       f : LinearMap (RingHom.id R) M₁ S
                       x✝ : M
                       x y : M₁
                       ⊢ Eq ((fun b => HSMul.hSMul (f b) x✝) (HAdd.hAdd x y)) (HAdd.hAdd ((fun b => H …
                     -/
  map_add' x y := by dsimp only; rw [f.map_add, add_smul]
                                 /-
                                   🎉 no goals
                                 -/
                      /-
                        R : Type u_1
                        R₂ : Type u_2
                        S : Type u_3
                        M : Type u_4
                        M₁ : Type u_5
                        M₂ : Type u_6
                        M₃ : Type u_7
                        N₁ : Type u_8
                        inst✝⁸ : Semiring R
                        inst✝⁷ : AddCommMonoid M
                        inst✝⁶ : AddCommMonoid M₁
                        inst✝⁵ : Module R M
                        inst✝⁴ : Module R M₁
                        inst✝³ : Semiring S
                        inst✝² : Module R S
                        inst✝¹ : Module S M
                        inst✝ : IsScalarTower R S M
                        f : LinearMap (RingHom.id R) M₁ S
                        x : M
                        b : R
                        y : M₁
                        ⊢ Eq ({ toFun := fun b => HSMul.hSMul (f b) x, map_add' := ⋯ }.toFun (HSMul.hS …
                      -/
  map_smul' b y := by dsimp; rw [map_smul, smul_assoc]
                             /-
                               🎉 no goals
                             -/


@[simp]
theorem coe_smulRight (f : M₁ →ₗ[R] S) (x : M) : (smulRight f x : M₁ → M) = fun c => f c • x :=
  rfl


theorem smulRight_apply (f : M₁ →ₗ[R] S) (x : M) (c : M₁) : smulRight f x c = f c • x :=
  rfl


@[simp]
                                                                      /-
                                                                        R : Type u_1
                                                                        S : Type u_3
                                                                        M : Type u_4
                                                                        M₁ : Type u_5
                                                                        inst✝⁸ : Semiring R
                                                                        inst✝⁷ : AddCommMonoid M
                                                                        inst✝⁶ : AddCommMonoid M₁
                                                                        inst✝⁵ : Module R M
                                                                        inst✝⁴ : Module R M₁
                                                                        inst✝³ : Semiring S
                                                                        inst✝² : Module R S
                                                                        inst✝¹ : Module S M
                                                                        inst✝ : IsScalarTower R S M
                                                                        f : LinearMap (RingHom.id R) M₁ S
                                                                        ⊢ Eq (f.smulRight 0) 0
                                                                      -/
lemma smulRight_zero (f : M₁ →ₗ[R] S) : f.smulRight (0 : M) = 0 := by ext; simp
                                                                           /-
                                                                             🎉 no goals
                                                                           -/


@[simp]
                                                                      /-
                                                                        R : Type u_1
                                                                        S : Type u_3
                                                                        M : Type u_4
                                                                        M₁ : Type u_5
                                                                        inst✝⁸ : Semiring R
                                                                        inst✝⁷ : AddCommMonoid M
                                                                        inst✝⁶ : AddCommMonoid M₁
                                                                        inst✝⁵ : Module R M
                                                                        inst✝⁴ : Module R M₁
                                                                        inst✝³ : Semiring S
                                                                        inst✝² : Module R S
                                                                        inst✝¹ : Module S M
                                                                        inst✝ : IsScalarTower R S M
                                                                        x : M
                                                                        ⊢ Eq (LinearMap.smulRight 0 x) 0
                                                                      -/
lemma zero_smulRight (x : M) : (0 : M₁ →ₗ[R] S).smulRight x = 0 := by ext; simp
                                                                           /-
                                                                             🎉 no goals
                                                                           -/


@[simp]
lemma smulRight_apply_eq_zero_iff {f : M₁ →ₗ[R] S} {x : M} [NoZeroSMulDivisors S M] :
    f.smulRight x = 0 ↔ f = 0 ∨ x = 0 := by
  /-
    R : Type u_1
    S : Type u_3
    M : Type u_4
    M₁ : Type u_5
    inst✝⁹ : Semiring R
    inst✝⁸ : AddCommMonoid M
    inst✝⁷ : AddCommMonoid M₁
    inst✝⁶ : Module R M
    inst✝⁵ : Module R M₁
    inst✝⁴ : Semiring S
    inst✝³ : Module R S
    inst✝² : Module S M
    inst✝¹ : IsScalarTower R S M
    f : LinearMap (RingHom.id R) M₁ S
    x : M
    inst✝ : NoZeroSMulDivisors S M
    ⊢ Iff (Eq (f.smulRight x) 0) (Or (Eq f 0) (Eq x 0))
  -/
  rcases eq_or_ne x 0 with rfl | hx
    /-
      case inl
      R : Type u_1
      S : Type u_3
      M : Type u_4
      M₁ : Type u_5
      inst✝⁹ : Semiring R
      inst✝⁸ : AddCommMonoid M
      inst✝⁷ : AddCommMonoid M₁
      inst✝⁶ : Module R M
      inst✝⁵ : Module R M₁
      inst✝⁴ : Semiring S
      inst✝³ : Module R S
      inst✝² : Module S M
      inst✝¹ : IsScalarTower R S M
      f : LinearMap (RingHom.id R) M₁ S
      inst✝ : NoZeroSMulDivisors S M
      ⊢ Iff (Eq (f.smulRight 0) 0) (Or (Eq f 0) (Eq 0 0))
    -/
  · simp
    /-
      🎉 no goals
    -/
  /-
    case inr
    R : Type u_1
    S : Type u_3
    M : Type u_4
    M₁ : Type u_5
    inst✝⁹ : Semiring R
    inst✝⁸ : AddCommMonoid M
    inst✝⁷ : AddCommMonoid M₁
    inst✝⁶ : Module R M
    inst✝⁵ : Module R M₁
    inst✝⁴ : Semiring S
    inst✝³ : Module R S
    inst✝² : Module S M
    inst✝¹ : IsScalarTower R S M
    f : LinearMap (RingHom.id R) M₁ S
    x : M
    inst✝ : NoZeroSMulDivisors S M
    hx : Ne x 0
    ⊢ Iff (Eq (f.smulRight x) 0) (Or (Eq f 0) (Eq x 0))
  -/
  refine ⟨fun h ↦ Or.inl ?_, fun h ↦ by simp [h.resolve_right hx]⟩
  /-
    case inr
    R : Type u_1
    S : Type u_3
    M : Type u_4
    M₁ : Type u_5
    inst✝⁹ : Semiring R
    inst✝⁸ : AddCommMonoid M
    inst✝⁷ : AddCommMonoid M₁
    inst✝⁶ : Module R M
    inst✝⁵ : Module R M₁
    inst✝⁴ : Semiring S
    inst✝³ : Module R S
    inst✝² : Module S M
    inst✝¹ : IsScalarTower R S M
    f : LinearMap (RingHom.id R) M₁ S
    x : M
    inst✝ : NoZeroSMulDivisors S M
    hx : Ne x 0
    h : Eq (f.smulRight x) 0
    ⊢ Eq f 0
  -/
  ext v
  /-
    case inr.h
    R : Type u_1
    S : Type u_3
    M : Type u_4
    M₁ : Type u_5
    inst✝⁹ : Semiring R
    inst✝⁸ : AddCommMonoid M
    inst✝⁷ : AddCommMonoid M₁
    inst✝⁶ : Module R M
    inst✝⁵ : Module R M₁
    inst✝⁴ : Semiring S
    inst✝³ : Module R S
    inst✝² : Module S M
    inst✝¹ : IsScalarTower R S M
    f : LinearMap (RingHom.id R) M₁ S
    x : M
    inst✝ : NoZeroSMulDivisors S M
    hx : Ne x 0
    h : Eq (f.smulRight x) 0
    v : M₁
    ⊢ Eq (f v) (0 v)
  -/
  replace h : f v • x = 0 := by simpa only [LinearMap.zero_apply] using LinearMap.congr_fun h v
  /-
    case inr.h
    R : Type u_1
    S : Type u_3
    M : Type u_4
    M₁ : Type u_5
    inst✝⁹ : Semiring R
    inst✝⁸ : AddCommMonoid M
    inst✝⁷ : AddCommMonoid M₁
    inst✝⁶ : Module R M
    inst✝⁵ : Module R M₁
    inst✝⁴ : Semiring S
    inst✝³ : Module R S
    inst✝² : Module S M
    inst✝¹ : IsScalarTower R S M
    f : LinearMap (RingHom.id R) M₁ S
    x : M
    inst✝ : NoZeroSMulDivisors S M
    hx : Ne x 0
    v : M₁
    h : Eq (HSMul.hSMul (f v) x) 0
    ⊢ Eq (f v) (0 v)
  -/
  rw [smul_eq_zero] at h
  /-
    case inr.h
    R : Type u_1
    S : Type u_3
    M : Type u_4
    M₁ : Type u_5
    inst✝⁹ : Semiring R
    inst✝⁸ : AddCommMonoid M
    inst✝⁷ : AddCommMonoid M₁
    inst✝⁶ : Module R M
    inst✝⁵ : Module R M₁
    inst✝⁴ : Semiring S
    inst✝³ : Module R S
    inst✝² : Module S M
    inst✝¹ : IsScalarTower R S M
    f : LinearMap (RingHom.id R) M₁ S
    x : M
    inst✝ : NoZeroSMulDivisors S M
    hx : Ne x 0
    v : M₁
    h : Or (Eq (f v) 0) (Eq x 0)
    ⊢ Eq (f v) (0 v)
  -/
  tauto
  /-
    🎉 no goals
  -/


/-- Applying a linear map at `v : M`, seen as `S`-linear map from `M →ₗ[R] M₂` to `M₂`.

 See `LinearMap.applyₗ` for a version where `S = R`. -/
@[simps]
def applyₗ' : M →+ (M →ₗ[R] M₂) →ₗ[S] M₂ where
  toFun v :=
    { toFun := fun f => f v
      map_add' := fun f g => f.add_apply g v
      map_smul' := fun x f => f.smul_apply x v }
  map_zero' := LinearMap.ext fun f => f.map_zero
  map_add' _ _ := LinearMap.ext fun f => f.map_add _ _


/-- Composition by `f : M₂ → M₃` is a linear map from the space of linear maps `M → M₂`
to the space of linear maps `M → M₃`. -/
def compRight (f : M₂ →ₗ[R] M₃) : (M →ₗ[R] M₂) →ₗ[R] M →ₗ[R] M₃ where
  toFun g := f.comp g
  map_add' _ _ := LinearMap.ext fun _ => map_add f _ _
  map_smul' _ _ := LinearMap.ext fun _ => map_smul f _ _


@[simp]
theorem compRight_apply (f : M₂ →ₗ[R] M₃) (g : M →ₗ[R] M₂) : compRight f g = f.comp g :=
  rfl


/-- Applying a linear map at `v : M`, seen as a linear map from `M →ₗ[R] M₂` to `M₂`.
See also `LinearMap.applyₗ'` for a version that works with two different semirings.

This is the `LinearMap` version of `toAddMonoidHom.eval`. -/
@[simps]
def applyₗ : M →ₗ[R] (M →ₗ[R] M₂) →ₗ[R] M₂ :=
  { applyₗ' R with
    toFun := fun v => { applyₗ' R v with toFun := fun f => f v }
    map_smul' := fun _ _ => LinearMap.ext fun f => map_smul f _ _ }


/--
The family of linear maps `M₂ → M` parameterised by `f ∈ M₂ → R`, `x ∈ M`, is linear in `f`, `x`.
-/
def smulRightₗ : (M₂ →ₗ[R] R) →ₗ[R] M →ₗ[R] M₂ →ₗ[R] M where
  toFun f :=
    { toFun := LinearMap.smulRight f
      map_add' := fun m m' => by
        /-
          R : Type u_1
          R₂ : Type u_2
          S : Type u_3
          M : Type u_4
          M₁ : Type u_5
          M₂ : Type u_6
          M₃ : Type u_7
          N₁ : Type u_8
          inst✝⁶ : CommSemiring R
          inst✝⁵ : AddCommMonoid M
          inst✝⁴ : AddCommMonoid M₂
          inst✝³ : AddCommMonoid M₃
          inst✝² : Module R M
          inst✝¹ : Module R M₂
          inst✝ : Module R M₃
          f✝ : LinearMap (RingHom.id R) M M₂
          f : LinearMap (RingHom.id R) M₂ R
          m m' : M
          ⊢ Eq (f.smulRight (HAdd.hAdd m m')) (HAdd.hAdd (f.smulRight m) (f.smulRight m'))
        -/
        ext
        /-
          case h
          R : Type u_1
          R₂ : Type u_2
          S : Type u_3
          M : Type u_4
          M₁ : Type u_5
          M₂ : Type u_6
          M₃ : Type u_7
          N₁ : Type u_8
          inst✝⁶ : CommSemiring R
          inst✝⁵ : AddCommMonoid M
          inst✝⁴ : AddCommMonoid M₂
          inst✝³ : AddCommMonoid M₃
          inst✝² : Module R M
          inst✝¹ : Module R M₂
          inst✝ : Module R M₃
          f✝ : LinearMap (RingHom.id R) M M₂
          f : LinearMap (RingHom.id R) M₂ R
          m m' : M
          x✝ : M₂
          ⊢ Eq ((f.smulRight (HAdd.hAdd m m')) x✝) ((HAdd.hAdd (f.smulRight m) (f.smulRi …
        -/
        apply smul_add
        /-
          🎉 no goals
        -/
      map_smul' := fun c m => by
        /-
          R : Type u_1
          R₂ : Type u_2
          S : Type u_3
          M : Type u_4
          M₁ : Type u_5
          M₂ : Type u_6
          M₃ : Type u_7
          N₁ : Type u_8
          inst✝⁶ : CommSemiring R
          inst✝⁵ : AddCommMonoid M
          inst✝⁴ : AddCommMonoid M₂
          inst✝³ : AddCommMonoid M₃
          inst✝² : Module R M
          inst✝¹ : Module R M₂
          inst✝ : Module R M₃
          f✝ : LinearMap (RingHom.id R) M M₂
          f : LinearMap (RingHom.id R) M₂ R
          c : R
          m : M
          ⊢ Eq ({ toFun := f.smulRight, map_add' := ⋯ }.toFun (HSMul.hSMul c m)) (HSMul. …
        -/
        ext
        /-
          case h
          R : Type u_1
          R₂ : Type u_2
          S : Type u_3
          M : Type u_4
          M₁ : Type u_5
          M₂ : Type u_6
          M₃ : Type u_7
          N₁ : Type u_8
          inst✝⁶ : CommSemiring R
          inst✝⁵ : AddCommMonoid M
          inst✝⁴ : AddCommMonoid M₂
          inst✝³ : AddCommMonoid M₃
          inst✝² : Module R M
          inst✝¹ : Module R M₂
          inst✝ : Module R M₃
          f✝ : LinearMap (RingHom.id R) M M₂
          f : LinearMap (RingHom.id R) M₂ R
          c : R
          m : M
          x✝ : M₂
          ⊢ Eq (({ toFun := f.smulRight, map_add' := ⋯ }.toFun (HSMul.hSMul c m)) x✝) (( …
        -/
        apply smul_comm }
        /-
          🎉 no goals
        -/
  map_add' f f' := by
    /-
      R : Type u_1
      R₂ : Type u_2
      S : Type u_3
      M : Type u_4
      M₁ : Type u_5
      M₂ : Type u_6
      M₃ : Type u_7
      N₁ : Type u_8
      inst✝⁶ : CommSemiring R
      inst✝⁵ : AddCommMonoid M
      inst✝⁴ : AddCommMonoid M₂
      inst✝³ : AddCommMonoid M₃
      inst✝² : Module R M
      inst✝¹ : Module R M₂
      inst✝ : Module R M₃
      f✝ : LinearMap (RingHom.id R) M M₂
      f f' : LinearMap (RingHom.id R) M₂ R
      ⊢ Eq ((fun f => { toFun := f.smulRight, map_add' := ⋯, map_smul' := ⋯ }) (HAdd …
    -/
    ext
    /-
      case h.h
      R : Type u_1
      R₂ : Type u_2
      S : Type u_3
      M : Type u_4
      M₁ : Type u_5
      M₂ : Type u_6
      M₃ : Type u_7
      N₁ : Type u_8
      inst✝⁶ : CommSemiring R
      inst✝⁵ : AddCommMonoid M
      inst✝⁴ : AddCommMonoid M₂
      inst✝³ : AddCommMonoid M₃
      inst✝² : Module R M
      inst✝¹ : Module R M₂
      inst✝ : Module R M₃
      f✝ : LinearMap (RingHom.id R) M M₂
      f f' : LinearMap (RingHom.id R) M₂ R
      x✝¹ : M
      x✝ : M₂
      ⊢ Eq ((((fun f => { toFun := f.smulRight, map_add' := ⋯, map_smul' := ⋯ }) (HA …
    -/
    apply add_smul
    /-
      🎉 no goals
    -/
  map_smul' c f := by
    /-
      R : Type u_1
      R₂ : Type u_2
      S : Type u_3
      M : Type u_4
      M₁ : Type u_5
      M₂ : Type u_6
      M₃ : Type u_7
      N₁ : Type u_8
      inst✝⁶ : CommSemiring R
      inst✝⁵ : AddCommMonoid M
      inst✝⁴ : AddCommMonoid M₂
      inst✝³ : AddCommMonoid M₃
      inst✝² : Module R M
      inst✝¹ : Module R M₂
      inst✝ : Module R M₃
      f✝ : LinearMap (RingHom.id R) M M₂
      c : R
      f : LinearMap (RingHom.id R) M₂ R
      ⊢ Eq ({ toFun := fun f => { toFun := f.smulRight, map_add' := ⋯, map_smul' :=  …
    -/
    ext
    /-
      case h.h
      R : Type u_1
      R₂ : Type u_2
      S : Type u_3
      M : Type u_4
      M₁ : Type u_5
      M₂ : Type u_6
      M₃ : Type u_7
      N₁ : Type u_8
      inst✝⁶ : CommSemiring R
      inst✝⁵ : AddCommMonoid M
      inst✝⁴ : AddCommMonoid M₂
      inst✝³ : AddCommMonoid M₃
      inst✝² : Module R M
      inst✝¹ : Module R M₂
      inst✝ : Module R M₃
      f✝ : LinearMap (RingHom.id R) M M₂
      c : R
      f : LinearMap (RingHom.id R) M₂ R
      x✝¹ : M
      x✝ : M₂
      ⊢ Eq ((({ toFun := fun f => { toFun := f.smulRight, map_add' := ⋯, map_smul' : …
    -/
    apply mul_smul
    /-
      🎉 no goals
    -/


@[simp]
theorem smulRightₗ_apply (f : M₂ →ₗ[R] R) (x : M) (c : M₂) :
    (smulRightₗ : (M₂ →ₗ[R] R) →ₗ[R] M →ₗ[R] M₂ →ₗ[R] M) f x c = f c • x :=
  rfl


