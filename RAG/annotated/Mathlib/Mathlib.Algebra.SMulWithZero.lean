/-- `SMulWithZero` is a class consisting of a Type `R` with `0 ∈ R` and a scalar multiplication
of `R` on a Type `M` with `0`, such that the equality `r • m = 0` holds if at least one among `r`
or `m` equals `0`. -/
class SMulWithZero [Zero R] [Zero M] extends SMulZeroClass R M where
  /-- Scalar multiplication by the scalar `0` is `0`. -/
  zero_smul : ∀ m : M, (0 : R) • m = 0


instance MulZeroClass.toSMulWithZero [MulZeroClass R] : SMulWithZero R R where
  smul := (· * ·)
  smul_zero := mul_zero
  zero_smul := zero_mul


/-- Like `MulZeroClass.toSMulWithZero`, but multiplies on the right. -/
instance MulZeroClass.toOppositeSMulWithZero [MulZeroClass R] : SMulWithZero Rᵐᵒᵖ R where
  smul := (· • ·)
  smul_zero _ := zero_mul _
  zero_smul := mul_zero


@[simp]
theorem zero_smul (m : M) : (0 : R) • m = 0 :=
  SMulWithZero.zero_smul m


lemma smul_eq_zero_of_left (h : a = 0) (b : M) : a • b = 0 := h.symm ▸ zero_smul _ b

lemma left_ne_zero_of_smul : a • b ≠ 0 → a ≠ 0 := mt fun h ↦ smul_eq_zero_of_left h b


/-- Pullback a `SMulWithZero` structure along an injective zero-preserving homomorphism.
See note [reducible non-instances]. -/
protected abbrev Function.Injective.smulWithZero (f : ZeroHom M' M) (hf : Function.Injective f)
    (smul : ∀ (a : R) (b), f (a • b) = a • f b) :
    SMulWithZero R M' where
  smul := (· • ·)
                          /-
                            R : Type u_1
                            R' : Type u_2
                            M : Type u_3
                            M' : Type u_4
                            inst✝⁵ : Zero R
                            inst✝⁴ : Zero M
                            inst✝³ : SMulWithZero R M
                            a✝ : R
                            b : M
                            inst✝² : Zero R'
                            inst✝¹ : Zero M'
                            inst✝ : SMul R M'
                            f : ZeroHom M' M
                            hf : Function.Injective ⇑f
                            smul : ∀ (a : R) (b : M'), Eq (f (HSMul.hSMul a b)) (HSMul.hSMul a (f b))
                            a : M'
                            ⊢ Eq (f (HSMul.hSMul 0 a)) (f 0)
                          -/
                          /-
                            R : Type u_1
                            R' : Type u_2
                            M : Type u_3
                            M' : Type u_4
                            inst✝⁵ : Zero R
                            inst✝⁴ : Zero M
                            inst✝³ : SMulWithZero R M
                            a✝ : R
                            b : M
                            inst✝² : Zero R'
                            inst✝¹ : Zero M'
                            inst✝ : SMul R M'
                            f : ZeroHom M' M
                            hf : Function.Injective ⇑f
                            smul : ∀ (a : R) (b : M'), Eq (f (HSMul.hSMul a b)) (HSMul.hSMul a (f b))
                            a : R
                            ⊢ Eq (f (HSMul.hSMul a 0)) (f 0)
                          -/
  zero_smul a := hf <| by simp [smul]
                          /-
                            🎉 no goals
                          -/
                          /-
                            🎉 no goals
                          -/
  smul_zero a := hf <| by simp [smul]


/-- Pushforward a `SMulWithZero` structure along a surjective zero-preserving homomorphism.
See note [reducible non-instances]. -/
protected abbrev Function.Surjective.smulWithZero (f : ZeroHom M M') (hf : Function.Surjective f)
    (smul : ∀ (a : R) (b), f (a • b) = a • f b) :
    SMulWithZero R M' where
  smul := (· • ·)
  zero_smul m := by
    /-
      R : Type u_1
      R' : Type u_2
      M : Type u_3
      M' : Type u_4
      inst✝⁵ : Zero R
      inst✝⁴ : Zero M
      inst✝³ : SMulWithZero R M
      a : R
      b : M
      inst✝² : Zero R'
      inst✝¹ : Zero M'
      inst✝ : SMul R M'
      f : ZeroHom M M'
      hf : Function.Surjective ⇑f
      smul : ∀ (a : R) (b : M), Eq (f (HSMul.hSMul a b)) (HSMul.hSMul a (f b))
      m : M'
      ⊢ Eq (HSMul.hSMul 0 m) 0
    -/
    rcases hf m with ⟨x, rfl⟩
                    /-
                      R : Type u_1
                      R' : Type u_2
                      M : Type u_3
                      M' : Type u_4
                      inst✝⁵ : Zero R
                      inst✝⁴ : Zero M
                      inst✝³ : SMulWithZero R M
                      a : R
                      b : M
                      inst✝² : Zero R'
                      inst✝¹ : Zero M'
                      inst✝ : SMul R M'
                      f : ZeroHom M M'
                      hf : Function.Surjective ⇑f
                      smul : ∀ (a : R) (b : M), Eq (f (HSMul.hSMul a b)) (HSMul.hSMul a (f b))
                      c : R
                      ⊢ Eq (HSMul.hSMul c 0) 0
                    -/
    /-
      case intro
      R : Type u_1
      R' : Type u_2
      M : Type u_3
      M' : Type u_4
      inst✝⁵ : Zero R
      inst✝⁴ : Zero M
      inst✝³ : SMulWithZero R M
      a : R
      b : M
      inst✝² : Zero R'
      inst✝¹ : Zero M'
      inst✝ : SMul R M'
      f : ZeroHom M M'
      hf : Function.Surjective ⇑f
      smul : ∀ (a : R) (b : M), Eq (f (HSMul.hSMul a b)) (HSMul.hSMul a (f b))
      x : M
      ⊢ Eq (HSMul.hSMul 0 (f x)) 0
    -/
                    /-
                      🎉 no goals
                    -/
    simp [← smul]
    /-
      🎉 no goals
    -/
  smul_zero c := by rw [← f.map_zero, ← smul, smul_zero]


/-- Compose a `SMulWithZero` with a `ZeroHom`, with action `f r' • m` -/
def SMulWithZero.compHom (f : ZeroHom R' R) : SMulWithZero R' M where
  smul := (f · • ·)
  smul_zero m := smul_zero (f m)
                    /-
                      R : Type u_1
                      R' : Type u_2
                      M : Type u_3
                      M' : Type u_4
                      inst✝⁵ : Zero R
                      inst✝⁴ : Zero M
                      inst✝³ : SMulWithZero R M
                      a : R
                      b : M
                      inst✝² : Zero R'
                      inst✝¹ : Zero M'
                      inst✝ : SMul R M'
                      f : ZeroHom R' R
                      m : M
                      ⊢ Eq (HSMul.hSMul 0 m) 0
                    -/
  zero_smul m := by show (f 0) • m = 0; rw [map_zero, zero_smul]
                                        /-
                                          🎉 no goals
                                        -/


instance AddMonoid.natSMulWithZero [AddMonoid M] : SMulWithZero ℕ M where
  smul_zero := _root_.nsmul_zero
  zero_smul := zero_nsmul


instance AddGroup.intSMulWithZero [AddGroup M] : SMulWithZero ℤ M where
  smul_zero := zsmul_zero
  zero_smul := zero_zsmul


/-- An action of a monoid with zero `R` on a Type `M`, also with `0`, extends `MulAction` and
is compatible with `0` (both in `R` and in `M`), with `1 ∈ R`, and with associativity of
multiplication on the monoid `M`. -/
class MulActionWithZero extends MulAction R M where
  -- these fields are copied from `SMulWithZero`, as `extends` behaves poorly
  /-- Scalar multiplication by any element send `0` to `0`. -/
  smul_zero : ∀ r : R, r • (0 : M) = 0
  /-- Scalar multiplication by the scalar `0` is `0`. -/
  zero_smul : ∀ m : M, (0 : R) • m = 0

-- see Note [lower instance priority]

instance (priority := 100) MulActionWithZero.toSMulWithZero
    (R M) {_ : MonoidWithZero R} {_ : Zero M} [m : MulActionWithZero R M] :
    SMulWithZero R M :=
  { m with }


/-- See also `Semiring.toModule` -/
instance MonoidWithZero.toMulActionWithZero : MulActionWithZero R R :=
  { MulZeroClass.toSMulWithZero R, Monoid.toMulAction R with }


/-- Like `MonoidWithZero.toMulActionWithZero`, but multiplies on the right. See also
`Semiring.toOppositeModule` -/
instance MonoidWithZero.toOppositeMulActionWithZero : MulActionWithZero Rᵐᵒᵖ R :=
  { MulZeroClass.toOppositeSMulWithZero R, Monoid.toOppositeMulAction with }


protected lemma MulActionWithZero.subsingleton
    [MulActionWithZero R M] [Subsingleton R] : Subsingleton M :=
  ⟨fun x y => by
    /-
      R : Type u_1
      M : Type u_3
      inst✝³ : MonoidWithZero R
      inst✝² : Zero M
      inst✝¹ : MulActionWithZero R M
      inst✝ : Subsingleton R
      x y : M
      ⊢ Eq x y
    -/
    rw [← one_smul R x, ← one_smul R y, Subsingleton.elim (1 : R) 0, zero_smul, zero_smul]⟩
    /-
      🎉 no goals
    -/


protected lemma MulActionWithZero.nontrivial
    [MulActionWithZero R M] [Nontrivial M] : Nontrivial R :=
  (subsingleton_or_nontrivial R).resolve_left fun _ =>
    not_subsingleton M <| MulActionWithZero.subsingleton R M


lemma ite_zero_smul (a : R) (b : M) : (if p then a else 0 : R) • b = if p then a • b else 0 := by
  /-
    R : Type u_1
    M : Type u_3
    inst✝³ : MonoidWithZero R
    inst✝² : Zero M
    inst✝¹ : MulActionWithZero R M
    p : Prop
    inst✝ : Decidable p
    a : R
    b : M
    ⊢ Eq (HSMul.hSMul (ite p a 0) b) (ite p (HSMul.hSMul a b) 0)
  -/
  rw [ite_smul, zero_smul]
  /-
    🎉 no goals
  -/


                                                                                   /-
                                                                                     R : Type u_1
                                                                                     M : Type u_3
                                                                                     inst✝³ : MonoidWithZero R
                                                                                     inst✝² : Zero M
                                                                                     inst✝¹ : MulActionWithZero R M
                                                                                     p : Prop
                                                                                     inst✝ : Decidable p
                                                                                     a : M
                                                                                     ⊢ Eq (HSMul.hSMul (ite p 1 0) a) (ite p a 0)
                                                                                   -/
lemma boole_smul (a : M) : (if p then 1 else 0 : R) • a = if p then a else 0 := by simp
                                                                                   /-
                                                                                     🎉 no goals
                                                                                   -/


lemma Pi.single_apply_smul {ι : Type*} [DecidableEq ι] (x : M) (i j : ι) :
    (Pi.single i 1 : ι → R) j • x = (Pi.single i x : ι → M) j := by
  /-
    R : Type u_1
    M : Type u_3
    inst✝³ : MonoidWithZero R
    inst✝² : Zero M
    inst✝¹ : MulActionWithZero R M
    ι : Type u_5
    inst✝ : DecidableEq ι
    x : M
    i j : ι
    ⊢ Eq (HSMul.hSMul (Pi.single i 1 j) x) (Pi.single i x j)
  -/
  rw [single_apply, ite_smul, one_smul, zero_smul, single_apply]
  /-
    🎉 no goals
  -/


/-- Pullback a `MulActionWithZero` structure along an injective zero-preserving homomorphism.
See note [reducible non-instances]. -/
protected abbrev Function.Injective.mulActionWithZero (f : ZeroHom M' M) (hf : Function.Injective f)
    (smul : ∀ (a : R) (b), f (a • b) = a • f b) : MulActionWithZero R M' :=
  { hf.mulAction f smul, hf.smulWithZero f smul with }


/-- Pushforward a `MulActionWithZero` structure along a surjective zero-preserving homomorphism.
See note [reducible non-instances]. -/
protected abbrev Function.Surjective.mulActionWithZero (f : ZeroHom M M')
    (hf : Function.Surjective f) (smul : ∀ (a : R) (b), f (a • b) = a • f b) :
    MulActionWithZero R M' :=
  { hf.mulAction f smul, hf.smulWithZero f smul with }


/-- Compose a `MulActionWithZero` with a `MonoidWithZeroHom`, with action `f r' • m` -/
def MulActionWithZero.compHom (f : R' →*₀ R) : MulActionWithZero R' M :=
  { SMulWithZero.compHom M f.toZeroHom with
                                /-
                                  R : Type u_1
                                  R' : Type u_2
                                  M : Type u_3
                                  M' : Type u_4
                                  inst✝⁶ : MonoidWithZero R
                                  inst✝⁵ : MonoidWithZero R'
                                  inst✝⁴ : Zero M
                                  inst✝³ : MulActionWithZero R M
                                  inst✝² : Zero M'
                                  inst✝¹ : SMul R M'
                                  p : Prop
                                  inst✝ : Decidable p
                                  f : MonoidWithZeroHom R' R
                                  r s : R'
                                  m : M
                                  ⊢ Eq (HSMul.hSMul (HMul.hMul r s) m) (HSMul.hSMul r (HSMul.hSMul s m))
                                -/
                            /-
                              R : Type u_1
                              R' : Type u_2
                              M : Type u_3
                              M' : Type u_4
                              inst✝⁶ : MonoidWithZero R
                              inst✝⁵ : MonoidWithZero R'
                              inst✝⁴ : Zero M
                              inst✝³ : MulActionWithZero R M
                              inst✝² : Zero M'
                              inst✝¹ : SMul R M'
                              p : Prop
                              inst✝ : Decidable p
                              f : MonoidWithZeroHom R' R
                              m : M
                              ⊢ Eq (HSMul.hSMul 1 m) m
                            -/
    mul_smul := fun r s m => by show f (r * s) • m = (f r) • (f s) • m; simp [mul_smul]
                                                /-
                                                  🎉 no goals
                                                -/
                                                                        /-
                                                                          🎉 no goals
                                                                        -/
    one_smul := fun m => by show (f 1) • m = m; simp }


theorem smul_inv₀ [SMulCommClass α β β] [IsScalarTower α β β] (c : α) (x : β) :
    (c • x)⁻¹ = c⁻¹ • x⁻¹ := by
  /-
    α : Type u_5
    β : Type u_6
    inst✝⁴ : GroupWithZero α
    inst✝³ : GroupWithZero β
    inst✝² : MulActionWithZero α β
    inst✝¹ : SMulCommClass α β β
    inst✝ : IsScalarTower α β β
    c : α
    x : β
    ⊢ Eq (Inv.inv (HSMul.hSMul c x)) (HSMul.hSMul (Inv.inv c) (Inv.inv x))
  -/
  obtain rfl | hc := eq_or_ne c 0
    /-
      case inl
      α : Type u_5
      β : Type u_6
      inst✝⁴ : GroupWithZero α
      inst✝³ : GroupWithZero β
      inst✝² : MulActionWithZero α β
      inst✝¹ : SMulCommClass α β β
      inst✝ : IsScalarTower α β β
      x : β
      ⊢ Eq (Inv.inv (HSMul.hSMul 0 x)) (HSMul.hSMul (Inv.inv 0) (Inv.inv x))
    -/
  · simp only [inv_zero, zero_smul]
    /-
      🎉 no goals
    -/
  /-
    case inr
    α : Type u_5
    β : Type u_6
    inst✝⁴ : GroupWithZero α
    inst✝³ : GroupWithZero β
    inst✝² : MulActionWithZero α β
    inst✝¹ : SMulCommClass α β β
    inst✝ : IsScalarTower α β β
    c : α
    x : β
    hc : Ne c 0
    ⊢ Eq (Inv.inv (HSMul.hSMul c x)) (HSMul.hSMul (Inv.inv c) (Inv.inv x))
  -/
  obtain rfl | hx := eq_or_ne x 0
    /-
      case inr.inl
      α : Type u_5
      β : Type u_6
      inst✝⁴ : GroupWithZero α
      inst✝³ : GroupWithZero β
      inst✝² : MulActionWithZero α β
      inst✝¹ : SMulCommClass α β β
      inst✝ : IsScalarTower α β β
      c : α
      hc : Ne c 0
      ⊢ Eq (Inv.inv (HSMul.hSMul c 0)) (HSMul.hSMul (Inv.inv c) (Inv.inv 0))
    -/
  · simp only [inv_zero, smul_zero]
    /-
      🎉 no goals
    -/
    /-
      case inr.inr
      α : Type u_5
      β : Type u_6
      inst✝⁴ : GroupWithZero α
      inst✝³ : GroupWithZero β
      inst✝² : MulActionWithZero α β
      inst✝¹ : SMulCommClass α β β
      inst✝ : IsScalarTower α β β
      c : α
      x : β
      hc : Ne c 0
      hx : Ne x 0
      ⊢ Eq (Inv.inv (HSMul.hSMul c x)) (HSMul.hSMul (Inv.inv c) (Inv.inv x))
    -/
  · refine inv_eq_of_mul_eq_one_left ?_
    /-
      case inr.inr
      α : Type u_5
      β : Type u_6
      inst✝⁴ : GroupWithZero α
      inst✝³ : GroupWithZero β
      inst✝² : MulActionWithZero α β
      inst✝¹ : SMulCommClass α β β
      inst✝ : IsScalarTower α β β
      c : α
      x : β
      hc : Ne c 0
      hx : Ne x 0
      ⊢ Eq (HMul.hMul (HSMul.hSMul (Inv.inv c) (Inv.inv x)) (HSMul.hSMul c x)) 1
    -/
    rw [smul_mul_smul_comm, inv_mul_cancel₀ hc, inv_mul_cancel₀ hx, one_smul]
    /-
      🎉 no goals
    -/


