/-- The `ZMod n`-module structure on commutative monoids whose elements have order dividing `n ≠ 0`.
Also implies a group structure via `Module.addCommMonoidToAddCommGroup`.
See note [reducible non-instances]. -/
abbrev AddCommMonoid.zmodModule [NeZero n] [AddCommMonoid M] (h : ∀ (x : M), n • x = 0) :
    Module (ZMod n) M := by
  have h_mod (c : ℕ) (x : M) : (c % n) • x = c • x := by
    suffices (c % n + c / n * n) • x = c • x by rwa [add_nsmul, mul_nsmul, h, add_zero] at this
    rw [Nat.mod_add_div']
  /-
    n : Nat
    M : Type u_1
    M₁ : Type u_2
    inst✝¹ : NeZero n
    inst✝ : AddCommMonoid M
    h : ∀ (x : M), Eq (HSMul.hSMul n x) 0
    h_mod : ∀ (c : Nat) (x : M), Eq (HSMul.hSMul (HMod.hMod c n) x) (HSMul.hSMul c …
    ⊢ Module (ZMod n) M
  -/
  have := NeZero.ne n
  match n with
  | n + 1 => exact {
    smul := fun (c : Fin _) x ↦ c.val • x
    smul_zero := fun _ ↦ nsmul_zero _
    zero_smul := fun _ ↦ zero_nsmul _
    smul_add := fun _ _ _ ↦ nsmul_add _ _ _
    one_smul := fun _ ↦ (h_mod _ _).trans <| one_nsmul _
    add_smul := fun _ _ _ ↦ (h_mod _ _).trans <| add_nsmul _ _ _
    mul_smul := fun _ _ _ ↦ (h_mod _ _).trans <| mul_nsmul' _ _ _
  }


/-- The `ZMod n`-module structure on Abelian groups whose elements have order dividing `n`.
See note [reducible non-instances]. -/
abbrev AddCommGroup.zmodModule {G : Type*} [AddCommGroup G] (h : ∀ (x : G), n • x = 0) :
    Module (ZMod n) G :=
  match n with
  | 0 => AddCommGroup.toIntModule G
  | _ + 1 => AddCommMonoid.zmodModule h


/-- The quotient of an abelian group by a subgroup containing all multiples of `n` is a
`n`-torsion group. -/
-- See note [reducible non-instances]
abbrev QuotientAddGroup.zmodModule {G : Type*} [AddCommGroup G] {H : AddSubgroup G}
    (hH : ∀ x, n • x ∈ H) : Module (ZMod n) (G ⧸ H) :=
                                /-
                                  n : Nat
                                  M : Type u_1
                                  M₁ : Type u_2
                                  G : Type u_3
                                  inst✝ : AddCommGroup G
                                  H : AddSubgroup G
                                  hH : ∀ (x : G), Membership.mem H (HSMul.hSMul n x)
                                  ⊢ ∀ (x : HasQuotient.Quotient G H), Eq (HSMul.hSMul n x) 0
                                -/
  AddCommGroup.zmodModule <| by simpa [QuotientAddGroup.forall_mk, ← QuotientAddGroup.mk_nsmul]
                                /-
                                  🎉 no goals
                                -/


theorem map_smul (f : F) (c : ZMod n) (x : M) : f (c • x) = c • f x := by
  /-
    n : Nat
    M : Type u_1
    M₁ : Type u_2
    F : Type u_3
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : AddCommGroup M₁
    inst✝³ : FunLike F M M₁
    inst✝² : AddMonoidHomClass F M M₁
    inst✝¹ : Module (ZMod n) M
    inst✝ : Module (ZMod n) M₁
    f : F
    c : ZMod n
    x : M
    ⊢ Eq (f (HSMul.hSMul c x)) (HSMul.hSMul c (f x))
  -/
  rw [← ZMod.intCast_zmod_cast c]
  /-
    n : Nat
    M : Type u_1
    M₁ : Type u_2
    F : Type u_3
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : AddCommGroup M₁
    inst✝³ : FunLike F M M₁
    inst✝² : AddMonoidHomClass F M M₁
    inst✝¹ : Module (ZMod n) M
    inst✝ : Module (ZMod n) M₁
    f : F
    c : ZMod n
    x : M
    ⊢ Eq (f (HSMul.hSMul (↑c.cast) x)) (HSMul.hSMul (↑c.cast) (f x))
  -/
  exact map_intCast_smul f _ _ (cast c) x
  /-
    🎉 no goals
  -/


theorem smul_mem (hx : x ∈ K) (c : ZMod n) : c • x ∈ K := by
  /-
    n : Nat
    M : Type u_1
    S : Type u_4
    inst✝³ : AddCommGroup M
    inst✝² : Module (ZMod n) M
    inst✝¹ : SetLike S M
    inst✝ : AddSubgroupClass S M
    x : M
    K : S
    hx : Membership.mem K x
    c : ZMod n
    ⊢ Membership.mem K (HSMul.hSMul c x)
  -/
  rw [← ZMod.intCast_zmod_cast c, Int.cast_smul_eq_zsmul]
  /-
    n : Nat
    M : Type u_1
    S : Type u_4
    inst✝³ : AddCommGroup M
    inst✝² : Module (ZMod n) M
    inst✝¹ : SetLike S M
    inst✝ : AddSubgroupClass S M
    x : M
    K : S
    hx : Membership.mem K x
    c : ZMod n
    ⊢ Membership.mem K (HSMul.hSMul c.cast x)
  -/
  exact zsmul_mem hx (cast c)
  /-
    🎉 no goals
  -/


/-- Reinterpret an additive homomorphism as a `ℤ/nℤ`-linear map.

See also:
`AddMonoidHom.toIntLinearMap`, `AddMonoidHom.toNatLinearMap`, `AddMonoidHom.toRatLinearMap` -/
def toZModLinearMap (f : M →+ M₁) : M →ₗ[ZMod n] M₁ := { f with map_smul' := ZMod.map_smul f }


theorem toZModLinearMap_injective : Function.Injective <| toZModLinearMap n (M := M) (M₁ := M₁) :=
  fun _ _ h ↦ ext fun x ↦ congr($h x)


@[simp]
theorem coe_toZModLinearMap (f : M →+ M₁) : ⇑(f.toZModLinearMap n) = f := rfl


/-- Reinterpret an additive subgroup of a `ℤ/nℤ`-module as a `ℤ/nℤ`-submodule.

See also: `AddSubgroup.toIntSubmodule`, `AddSubmonoid.toNatSubmodule`. -/
def toZModSubmodule : AddSubgroup M ≃o Submodule (ZMod n) M where
  toFun S := { S with smul_mem' := fun c _ h ↦ ZMod.smul_mem (K := S) h c }
  invFun := Submodule.toAddSubgroup
  left_inv _ := rfl
  right_inv _ := rfl
  map_rel_iff' := Iff.rfl


@[simp]
theorem toZModSubmodule_symm :
    ⇑((toZModSubmodule n).symm : _ ≃o AddSubgroup M) = Submodule.toAddSubgroup :=
  rfl


@[simp] lemma coe_toZModSubmodule (S : AddSubgroup M) : (toZModSubmodule n S : Set M) = S := rfl

@[simp] lemma mem_toZModSubmodule {S : AddSubgroup M} : x ∈ toZModSubmodule n S ↔ x ∈ S := .rfl


@[simp]
theorem toZModSubmodule_toAddSubgroup (S : AddSubgroup M) :
    (toZModSubmodule n S).toAddSubgroup = S :=
  rfl


@[simp]
theorem _root_.Submodule.toAddSubgroup_toZModSubmodule (S : Submodule (ZMod n) M) :
    toZModSubmodule n S.toAddSubgroup = S :=
  rfl


