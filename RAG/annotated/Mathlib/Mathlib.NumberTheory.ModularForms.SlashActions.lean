/-- A general version of the slash action of the space of modular forms. -/
class SlashAction (β G α γ : Type*) [Group G] [AddMonoid α] [SMul γ α] where
  map : β → G → α → α
  zero_slash : ∀ (k : β) (g : G), map k g 0 = 0
  slash_one : ∀ (k : β) (a : α), map k 1 a = a
  slash_mul : ∀ (k : β) (g h : G) (a : α), map k (g * h) a = map k h (map k g a)
  smul_slash : ∀ (k : β) (g : G) (a : α) (z : γ), map k g (z • a) = z • map k g a
  add_slash : ∀ (k : β) (g : G) (a b : α), map k g (a + b) = map k g a + map k g b


scoped[ModularForm] notation:100 f " ∣[" k ";" γ "] " a:100 => SlashAction.map γ k a f


scoped[ModularForm] notation:100 f " ∣[" k "] " a:100 => SlashAction.map ℂ k a f


@[simp]
theorem SlashAction.neg_slash {β G α γ : Type*} [Group G] [AddGroup α] [SMul γ α]
    [SlashAction β G α γ] (k : β) (g : G) (a : α) : (-a) ∣[k;γ] g = -a ∣[k;γ] g :=
  eq_neg_of_add_eq_zero_left <| by
    /-
      β : Type u_1
      G : Type u_2
      α : Type u_3
      γ : Type u_4
      inst✝³ : Group G
      inst✝² : AddGroup α
      inst✝¹ : SMul γ α
      inst✝ : SlashAction β G α γ
      k : β
      g : G
      a : α
      ⊢ Eq (HAdd.hAdd (SlashAction.map γ k g (Neg.neg a)) (SlashAction.map γ k g a)) 0
    -/
    rw [← SlashAction.add_slash, neg_add_cancel, SlashAction.zero_slash]
    /-
      🎉 no goals
    -/


@[simp]
theorem SlashAction.smul_slash_of_tower {R β G α : Type*} (γ : Type*) [Group G] [AddGroup α]
    [Monoid γ] [MulAction γ α] [SMul R γ] [SMul R α] [IsScalarTower R γ α] [SlashAction β G α γ]
    (k : β) (g : G) (a : α) (r : R) : (r • a) ∣[k;γ] g = r • a ∣[k;γ] g := by
  /-
    R : Type u_1
    β : Type u_2
    G : Type u_3
    α : Type u_4
    γ : Type u_5
    inst✝⁷ : Group G
    inst✝⁶ : AddGroup α
    inst✝⁵ : Monoid γ
    inst✝⁴ : MulAction γ α
    inst✝³ : SMul R γ
    inst✝² : SMul R α
    inst✝¹ : IsScalarTower R γ α
    inst✝ : SlashAction β G α γ
    k : β
    g : G
    a : α
    r : R
    ⊢ Eq (SlashAction.map γ k g (HSMul.hSMul r a)) (HSMul.hSMul r (SlashAction.map …
  -/
  rw [← smul_one_smul γ r a, SlashAction.smul_slash, smul_one_smul]
  /-
    🎉 no goals
  -/


/-- Slash_action induced by a monoid homomorphism. -/
def monoidHomSlashAction {β G H α γ : Type*} [Group G] [AddMonoid α] [SMul γ α] [Group H]
    [SlashAction β G α γ] (h : H →* G) : SlashAction β H α γ where
  map k g := SlashAction.map γ k (h g)
  zero_slash k g := SlashAction.zero_slash k (h g)
                      /-
                        β : Type u_1
                        G : Type u_2
                        H : Type u_3
                        α : Type u_4
                        γ : Type u_5
                        inst✝⁴ : Group G
                        inst✝³ : AddMonoid α
                        inst✝² : SMul γ α
                        inst✝¹ : Group H
                        inst✝ : SlashAction β G α γ
                        h : MonoidHom H G
                        k : β
                        a : α
                        ⊢ Eq ((fun k g => SlashAction.map γ k (h g)) k 1 a) a
                      -/
  slash_one k a := by simp only [map_one, SlashAction.slash_one]
                      /-
                        🎉 no goals
                      -/
                           /-
                             β : Type u_1
                             G : Type u_2
                             H : Type u_3
                             α : Type u_4
                             γ : Type u_5
                             inst✝⁴ : Group G
                             inst✝³ : AddMonoid α
                             inst✝² : SMul γ α
                             inst✝¹ : Group H
                             inst✝ : SlashAction β G α γ
                             h : MonoidHom H G
                             k : β
                             g gg : H
                             a : α
                             ⊢ Eq ((fun k g => SlashAction.map γ k (h g)) k (HMul.hMul g gg) a) ((fun k g = …
                           -/
  slash_mul k g gg a := by simp only [map_mul, SlashAction.slash_mul]
                           /-
                             🎉 no goals
                           -/
  smul_slash _ _ := SlashAction.smul_slash _ _
  add_slash _ g _ _ := SlashAction.add_slash _ (h g) _ _


/-- The weight `k` action of `GL(2, ℝ)⁺` on functions `f : ℍ → ℂ`. -/
def slash (k : ℤ) (γ : GL(2, ℝ)⁺) (f : ℍ → ℂ) (x : ℍ) : ℂ :=
  f (γ • x) * (↑(↑ₘ[ℝ] γ).det : ℂ) ^ (k - 1) * UpperHalfPlane.denom γ x ^ (-k)


local notation:100 f " ∣[" k "]" γ:100 => ModularForm.slash k γ f


private theorem slash_mul (k : ℤ) (A B : GL(2, ℝ)⁺) (f : ℍ → ℂ) :
    f ∣[k] (A * B) = (f ∣[k] A) ∣[k] B := by
  /-
    k : Int
    A B : Subtype fun x => Membership.mem (Matrix.GLPos (Fin 2) Real) x
    f : UpperHalfPlane → Complex
    ⊢ Eq (ModularForm.slash k (HMul.hMul A B) f) (ModularForm.slash k B (ModularFo …
  -/
  ext1 x
  /-
    case h
    k : Int
    A B : Subtype fun x => Membership.mem (Matrix.GLPos (Fin 2) Real) x
    f : UpperHalfPlane → Complex
    x : UpperHalfPlane
    ⊢ Eq (ModularForm.slash k (HMul.hMul A B) f x) (ModularForm.slash k B (Modular …
  -/
  simp only [slash, UpperHalfPlane.denom_cocycle A B x]
  simp only [mul_smul, Subgroup.coe_mul, Units.val_mul, Matrix.det_mul, ofReal_mul, denom, smulAux,
    smulAux', num, coe_mk, UpperHalfPlane.coe_smul]
  rw [mul_zpow, mul_right_comm _ _ (((↑ₘ[ℝ] B).det : ℂ) ^ (k - 1)),
    ← mul_assoc, mul_zpow, ← mul_assoc]


private theorem add_slash (k : ℤ) (A : GL(2, ℝ)⁺) (f g : ℍ → ℂ) :
    (f + g) ∣[k]A = f ∣[k]A + g ∣[k]A := by
  /-
    k : Int
    A : Subtype fun x => Membership.mem (Matrix.GLPos (Fin 2) Real) x
    f g : UpperHalfPlane → Complex
    ⊢ Eq (ModularForm.slash k A (HAdd.hAdd f g)) (HAdd.hAdd (ModularForm.slash k A …
  -/
  ext1
  /-
    case h
    k : Int
    A : Subtype fun x => Membership.mem (Matrix.GLPos (Fin 2) Real) x
    f g : UpperHalfPlane → Complex
    x✝ : UpperHalfPlane
    ⊢ Eq (ModularForm.slash k A (HAdd.hAdd f g) x✝) (HAdd.hAdd (ModularForm.slash  …
  -/
  simp only [slash, Pi.add_apply, denom, zpow_neg]
  /-
    case h
    k : Int
    A : Subtype fun x => Membership.mem (Matrix.GLPos (Fin 2) Real) x
    f g : UpperHalfPlane → Complex
    x✝ : UpperHalfPlane
    ⊢ Eq (HMul.hMul (HMul.hMul (HAdd.hAdd (f (HSMul.hSMul A x✝)) (g (HSMul.hSMul A …
  -/
  ring
  /-
    🎉 no goals
  -/


private theorem slash_one (k : ℤ) (f : ℍ → ℂ) : f ∣[k]1 = f :=
               /-
                 k : Int
                 f : UpperHalfPlane → Complex
                 ⊢ ∀ (x : UpperHalfPlane), Eq (ModularForm.slash k 1 f x) (f x)
               -/
  funext <| by simp [slash, denom]
               /-
                 🎉 no goals
               -/


private theorem smul_slash (k : ℤ) (A : GL(2, ℝ)⁺) (f : ℍ → ℂ) (c : α) :
    (c • f) ∣[k]A = c • f ∣[k]A := by
  /-
    α : Type u_1
    inst✝¹ : SMul α Complex
    inst✝ : IsScalarTower α Complex Complex
    k : Int
    A : Subtype fun x => Membership.mem (Matrix.GLPos (Fin 2) Real) x
    f : UpperHalfPlane → Complex
    c : α
    ⊢ Eq (ModularForm.slash k A (HSMul.hSMul c f)) (HSMul.hSMul c (ModularForm.sla …
  -/
  simp_rw [← smul_one_smul ℂ c f, ← smul_one_smul ℂ c (f ∣[k]A)]
  /-
    α : Type u_1
    inst✝¹ : SMul α Complex
    inst✝ : IsScalarTower α Complex Complex
    k : Int
    A : Subtype fun x => Membership.mem (Matrix.GLPos (Fin 2) Real) x
    f : UpperHalfPlane → Complex
    c : α
    ⊢ Eq (ModularForm.slash k A (HSMul.hSMul (HSMul.hSMul c 1) f)) (HSMul.hSMul (H …
  -/
  ext1
  /-
    case h
    α : Type u_1
    inst✝¹ : SMul α Complex
    inst✝ : IsScalarTower α Complex Complex
    k : Int
    A : Subtype fun x => Membership.mem (Matrix.GLPos (Fin 2) Real) x
    f : UpperHalfPlane → Complex
    c : α
    x✝ : UpperHalfPlane
    ⊢ Eq (ModularForm.slash k A (HSMul.hSMul (HSMul.hSMul c 1) f) x✝) (HSMul.hSMul …
  -/
  simp_rw [slash]
  /-
    case h
    α : Type u_1
    inst✝¹ : SMul α Complex
    inst✝ : IsScalarTower α Complex Complex
    k : Int
    A : Subtype fun x => Membership.mem (Matrix.GLPos (Fin 2) Real) x
    f : UpperHalfPlane → Complex
    c : α
    x✝ : UpperHalfPlane
    ⊢ Eq (HMul.hMul (HMul.hMul (HSMul.hSMul (HSMul.hSMul c 1) f (HSMul.hSMul A x✝) …
  -/
  simp only [slash, Algebra.id.smul_eq_mul, Matrix.GeneralLinearGroup.val_det_apply, Pi.smul_apply]
  /-
    case h
    α : Type u_1
    inst✝¹ : SMul α Complex
    inst✝ : IsScalarTower α Complex Complex
    k : Int
    A : Subtype fun x => Membership.mem (Matrix.GLPos (Fin 2) Real) x
    f : UpperHalfPlane → Complex
    c : α
    x✝ : UpperHalfPlane
    ⊢ Eq (HMul.hMul (HMul.hMul (HMul.hMul (HSMul.hSMul c 1) (f (HSMul.hSMul A x✝)) …
  -/
  ring
  /-
    🎉 no goals
  -/


private theorem zero_slash (k : ℤ) (A : GL(2, ℝ)⁺) : (0 : ℍ → ℂ) ∣[k]A = 0 :=
                     /-
                       k : Int
                       A : Subtype fun x => Membership.mem (Matrix.GLPos (Fin 2) Real) x
                       x✝ : UpperHalfPlane
                       ⊢ Eq (ModularForm.slash k A 0 x✝) (0 x✝)
                     -/
  funext fun _ => by simp only [slash, Pi.zero_apply, zero_mul]
                     /-
                       🎉 no goals
                     -/


instance : SlashAction ℤ GL(2, ℝ)⁺ (ℍ → ℂ) ℂ where
  map := slash
  zero_slash := zero_slash
  slash_one := slash_one
  slash_mul := slash_mul
  smul_slash := smul_slash
  add_slash := add_slash


theorem slash_def (A : GL(2, ℝ)⁺) : f ∣[k] A = slash k A f :=
  rfl


instance SLAction : SlashAction ℤ SL(2, ℤ) (ℍ → ℂ) ℂ :=
  monoidHomSlashAction
    (MonoidHom.comp Matrix.SpecialLinearGroup.toGLPos
      (Matrix.SpecialLinearGroup.map (Int.castRingHom ℝ)))


@[simp]
theorem SL_slash (γ : SL(2, ℤ)) : f ∣[k] γ = f ∣[k] (γ : GL(2, ℝ)⁺) :=
  rfl


theorem is_invariant_const (A : SL(2, ℤ)) (x : ℂ) :
    Function.const ℍ x ∣[(0 : ℤ)] A = Function.const ℍ x := by
  /-
    A : Matrix.SpecialLinearGroup (Fin 2) Int
    x : Complex
    ⊢ Eq (SlashAction.map Complex 0 A (Function.const UpperHalfPlane x)) (Function …
  -/
  funext
  simp only [SL_slash, slash_def, slash, Function.const_apply, det_coe, ofReal_one, zero_sub,
    zpow_neg, zpow_one, inv_one, mul_one, neg_zero, zpow_zero]


/-- The constant function 1 is invariant under any element of `SL(2, ℤ)`. -/
theorem is_invariant_one (A : SL(2, ℤ)) : (1 : ℍ → ℂ) ∣[(0 : ℤ)] A = (1 : ℍ → ℂ) :=
  is_invariant_const _ _


/-- Variant of `is_invariant_one` with the left hand side in simp normal form. -/
@[simp]
theorem is_invariant_one' (A : SL(2, ℤ)) : (1 : ℍ → ℂ) ∣[(0 : ℤ)] (A : GL(2, ℝ)⁺) = 1 := by
  /-
    A : Matrix.SpecialLinearGroup (Fin 2) Int
    ⊢ Eq (SlashAction.map Complex 0 (↑A) 1) 1
  -/
  simpa using is_invariant_one A
  /-
    🎉 no goals
  -/


/-- A function `f : ℍ → ℂ` is slash-invariant, of weight `k ∈ ℤ` and level `Γ`,
  if for every matrix `γ ∈ Γ` we have `f(γ • z)= (c*z+d)^k f(z)` where `γ= ![![a, b], ![c, d]]`,
  and it acts on `ℍ` via Möbius transformations. -/
theorem slash_action_eq'_iff (k : ℤ) (f : ℍ → ℂ) (γ : SL(2, ℤ)) (z : ℍ) :
    (f ∣[k] γ) z = f z ↔ f (γ • z) = ((γ 1 0 : ℂ) * z + (γ 1 1 : ℂ)) ^ k * f z := by
  /-
    k : Int
    f : UpperHalfPlane → Complex
    γ : Matrix.SpecialLinearGroup (Fin 2) Int
    z : UpperHalfPlane
    ⊢ Iff (Eq (SlashAction.map Complex k γ f z) (f z)) (Eq (f (HSMul.hSMul γ z)) ( …
  -/
  simp only [SL_slash, slash_def, ModularForm.slash]
  /-
    k : Int
    f : UpperHalfPlane → Complex
    γ : Matrix.SpecialLinearGroup (Fin 2) Int
    z : UpperHalfPlane
    ⊢ Iff (Eq (HMul.hMul (HMul.hMul (f (HSMul.hSMul (↑γ) z)) (HPow.hPow (↑(↑↑↑γ).d …
  -/
  convert inv_mul_eq_iff_eq_mul₀ (G₀ := ℂ) _ using 2
    /-
      case h.e'_1.h.e'_2
      k : Int
      f : UpperHalfPlane → Complex
      γ : Matrix.SpecialLinearGroup (Fin 2) Int
      z : UpperHalfPlane
      ⊢ Eq (HMul.hMul (HMul.hMul (f (HSMul.hSMul (↑γ) z)) (HPow.hPow (↑(↑↑↑γ).det) ( …
    -/
  · rw [mul_comm]
    simp only [denom, zpow_neg, det_coe, ofReal_one, one_zpow, mul_one,
      sl_moeb]
    /-
      case h.e'_1.h.e'_2
      k : Int
      f : UpperHalfPlane → Complex
      γ : Matrix.SpecialLinearGroup (Fin 2) Int
      z : UpperHalfPlane
      ⊢ Eq (HMul.hMul (Inv.inv (HPow.hPow (HAdd.hAdd (HMul.hMul ↑(↑↑↑γ 1 0) ↑z) ↑(↑↑ …
    -/
    rfl
    /-
      🎉 no goals
    -/
    /-
      case convert_4
      k : Int
      f : UpperHalfPlane → Complex
      γ : Matrix.SpecialLinearGroup (Fin 2) Int
      z : UpperHalfPlane
      ⊢ Ne (HPow.hPow (HAdd.hAdd (HMul.hMul ↑(↑γ 1 0) ↑z) ↑(↑γ 1 1)) k) 0
    -/
  · convert zpow_ne_zero k (denom_ne_zero γ z)
    /-
      🎉 no goals
    -/


theorem mul_slash (k1 k2 : ℤ) (A : GL(2, ℝ)⁺) (f g : ℍ → ℂ) :
    (f * g) ∣[k1 + k2] A = ((↑ₘA).det : ℝ) • f ∣[k1] A * g ∣[k2] A := by
  /-
    k1 k2 : Int
    A : Subtype fun x => Membership.mem (Matrix.GLPos (Fin 2) Real) x
    f g : UpperHalfPlane → Complex
    ⊢ Eq (SlashAction.map Complex (HAdd.hAdd k1 k2) A (HMul.hMul f g)) (HMul.hMul  …
  -/
  ext1 x
  simp only [slash_def, slash, Matrix.GeneralLinearGroup.val_det_apply,
    Pi.mul_apply, Pi.smul_apply, Algebra.smul_mul_assoc, real_smul]
  /-
    case h
    k1 k2 : Int
    A : Subtype fun x => Membership.mem (Matrix.GLPos (Fin 2) Real) x
    f g : UpperHalfPlane → Complex
    x : UpperHalfPlane
    ⊢ Eq (HMul.hMul (HMul.hMul (HMul.hMul (f (HSMul.hSMul A x)) (g (HSMul.hSMul A  …
  -/
  set d : ℂ := ↑(↑ₘ[ℝ] A).det
  have h1 : d ^ (k1 + k2 - 1) = d * d ^ (k1 - 1) * d ^ (k2 - 1) := by
    have : d ≠ 0 := by
      dsimp only [d]
      exact_mod_cast Matrix.GLPos.det_ne_zero A
    rw [← zpow_one_add₀ this, ← zpow_add₀ this]
    congr; ring
  have h22 : denom A x ^ (-(k1 + k2)) = denom A x ^ (-k1) * denom A x ^ (-k2) := by
    rw [Int.neg_add, zpow_add₀]
    exact UpperHalfPlane.denom_ne_zero A x
  /-
    case h
    k1 k2 : Int
    A : Subtype fun x => Membership.mem (Matrix.GLPos (Fin 2) Real) x
    f g : UpperHalfPlane → Complex
    x : UpperHalfPlane
    d : Complex := ↑(↑↑A).det
    h1 : Eq (HPow.hPow d (HSub.hSub (HAdd.hAdd k1 k2) 1)) (HMul.hMul (HMul.hMul d  …
    h22 : Eq (HPow.hPow (UpperHalfPlane.denom A x) (Neg.neg (HAdd.hAdd k1 k2))) (H …
    ⊢ Eq (HMul.hMul (HMul.hMul (HMul.hMul (f (HSMul.hSMul A x)) (g (HSMul.hSMul A  …
  -/
  rw [h1, h22]
  /-
    case h
    k1 k2 : Int
    A : Subtype fun x => Membership.mem (Matrix.GLPos (Fin 2) Real) x
    f g : UpperHalfPlane → Complex
    x : UpperHalfPlane
    d : Complex := ↑(↑↑A).det
    h1 : Eq (HPow.hPow d (HSub.hSub (HAdd.hAdd k1 k2) 1)) (HMul.hMul (HMul.hMul d  …
    h22 : Eq (HPow.hPow (UpperHalfPlane.denom A x) (Neg.neg (HAdd.hAdd k1 k2))) (H …
    ⊢ Eq (HMul.hMul (HMul.hMul (HMul.hMul (f (HSMul.hSMul A x)) (g (HSMul.hSMul A  …
  -/
  ring
  /-
    🎉 no goals
  -/


theorem mul_slash_SL2 (k1 k2 : ℤ) (A : SL(2, ℤ)) (f g : ℍ → ℂ) :
    (f * g) ∣[k1 + k2] A = f ∣[k1] A * g ∣[k2] A :=
  calc
    (f * g) ∣[k1 + k2] (A : GL(2, ℝ)⁺) =
        ((↑ₘA).det : ℝ) • f ∣[k1] A * g ∣[k2] A := by
      /-
        k1 k2 : Int
        A : Matrix.SpecialLinearGroup (Fin 2) Int
        f g : UpperHalfPlane → Complex
        ⊢ Eq (SlashAction.map Complex (HAdd.hAdd k1 k2) (↑A) (HMul.hMul f g)) (HMul.hM …
      -/
      apply mul_slash
      /-
        🎉 no goals
      -/
                                              /-
                                                k1 k2 : Int
                                                A : Matrix.SpecialLinearGroup (Fin 2) Int
                                                f g : UpperHalfPlane → Complex
                                                ⊢ Eq (HMul.hMul (HSMul.hSMul (↑↑↑A).det (SlashAction.map Complex k1 A f)) (Sla …
                                              -/
    _ = (1 : ℝ) • f ∣[k1] A * g ∣[k2] A := by rw [det_coe]
                                              /-
                                                🎉 no goals
                                              -/
                                    /-
                                      k1 k2 : Int
                                      A : Matrix.SpecialLinearGroup (Fin 2) Int
                                      f g : UpperHalfPlane → Complex
                                      ⊢ Eq (HMul.hMul (HSMul.hSMul 1 (SlashAction.map Complex k1 A f)) (SlashAction. …
                                    -/
    _ = f ∣[k1] A * g ∣[k2] A := by rw [one_smul]
                                    /-
                                      🎉 no goals
                                    -/


