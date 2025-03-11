/-- Complex numbers consist of two `Real`s: a real part `re` and an imaginary part `im`. -/
structure Complex : Type where
  /-- The real part of a complex number. -/
  re : ℝ
  /-- The imaginary part of a complex number. -/
  im : ℝ


@[inherit_doc] notation "ℂ" => Complex


noncomputable instance : DecidableEq ℂ :=
  Classical.decEq _


/-- The equivalence between the complex numbers and `ℝ × ℝ`. -/
@[simps apply]
def equivRealProd : ℂ ≃ ℝ × ℝ where
  toFun z := ⟨z.re, z.im⟩
  invFun p := ⟨p.1, p.2⟩
  left_inv := fun ⟨_, _⟩ => rfl
  right_inv := fun ⟨_, _⟩ => rfl


@[simp]
theorem eta : ∀ z : ℂ, Complex.mk z.re z.im = z
  | ⟨_, _⟩ => rfl

-- We only mark this lemma with `ext` *locally* to avoid it applying whenever terms of `ℂ` appear.

theorem ext : ∀ {z w : ℂ}, z.re = w.re → z.im = w.im → z = w
  | ⟨_, _⟩, ⟨_, _⟩, rfl, rfl => rfl


attribute [local ext] Complex.ext


                                                                   /-
                                                                     p : Complex → Prop
                                                                     ⊢ Iff (∀ (x : Complex), p x) (∀ (a b : Real), p { re := a, im := b })
                                                                   -/
lemma «forall» {p : ℂ → Prop} : (∀ x, p x) ↔ ∀ a b, p ⟨a, b⟩ := by aesop
                                                                   /-
                                                                     🎉 no goals
                                                                   -/

                                                                   /-
                                                                     p : Complex → Prop
                                                                     ⊢ Iff (Exists fun x => p x) (Exists fun a => Exists fun b => p { re := a, im : …
                                                                   -/
lemma «exists» {p : ℂ → Prop} : (∃ x, p x) ↔ ∃ a b, p ⟨a, b⟩ := by aesop
                                                                   /-
                                                                     🎉 no goals
                                                                   -/


theorem re_surjective : Surjective re := fun x => ⟨⟨x, 0⟩, rfl⟩


theorem im_surjective : Surjective im := fun y => ⟨⟨0, y⟩, rfl⟩


@[simp]
theorem range_re : range re = univ :=
  re_surjective.range_eq


@[simp]
theorem range_im : range im = univ :=
  im_surjective.range_eq

-- Porting note: refactored instance to allow `norm_cast` to work

/-- The natural inclusion of the real numbers into the complex numbers. -/
@[coe]
def ofReal (r : ℝ) : ℂ :=
  ⟨r, 0⟩

instance : Coe ℝ ℂ :=
  ⟨ofReal⟩


@[deprecated (since := "2024-10-12")] alias ofReal' := ofReal


@[simp, norm_cast]
theorem ofReal_re (r : ℝ) : Complex.re (r : ℂ) = r :=
  rfl


@[simp, norm_cast]
theorem ofReal_im (r : ℝ) : (r : ℂ).im = 0 :=
  rfl


theorem ofReal_def (r : ℝ) : (r : ℂ) = ⟨r, 0⟩ :=
  rfl


@[simp, norm_cast]
theorem ofReal_inj {z w : ℝ} : (z : ℂ) = w ↔ z = w :=
                   /-
                     z w : Real
                     ⊢ Eq z w → Eq ↑z ↑w
                   -/
  ⟨congrArg re, by apply congrArg⟩
                   /-
                     🎉 no goals
                   -/

-- Porting note: made coercion explicit

theorem ofReal_injective : Function.Injective ((↑) : ℝ → ℂ) := fun _ _ => congrArg re

-- Porting note: made coercion explicit

instance canLift : CanLift ℂ ℝ (↑) fun z => z.im = 0 where
  prf z hz := ⟨z.re, ext rfl hz.symm⟩


/-- The product of a set on the real axis and a set on the imaginary axis of the complex plane,
denoted by `s ×ℂ t`. -/
def reProdIm (s t : Set ℝ) : Set ℂ :=
  re ⁻¹' s ∩ im ⁻¹' t


@[deprecated (since := "2024-12-03")] protected alias Set.reProdIm := reProdIm


@[inherit_doc]
infixl:72 " ×ℂ " => reProdIm


theorem mem_reProdIm {z : ℂ} {s t : Set ℝ} : z ∈ s ×ℂ t ↔ z.re ∈ s ∧ z.im ∈ t :=
  Iff.rfl


instance : Zero ℂ :=
  ⟨(0 : ℝ)⟩


instance : Inhabited ℂ :=
  ⟨0⟩


@[simp]
theorem zero_re : (0 : ℂ).re = 0 :=
  rfl


@[simp]
theorem zero_im : (0 : ℂ).im = 0 :=
  rfl


@[simp, norm_cast]
theorem ofReal_zero : ((0 : ℝ) : ℂ) = 0 :=
  rfl


@[simp]
theorem ofReal_eq_zero {z : ℝ} : (z : ℂ) = 0 ↔ z = 0 :=
  ofReal_inj


theorem ofReal_ne_zero {z : ℝ} : (z : ℂ) ≠ 0 ↔ z ≠ 0 :=
  not_congr ofReal_eq_zero


instance : One ℂ :=
  ⟨(1 : ℝ)⟩


@[simp]
theorem one_re : (1 : ℂ).re = 1 :=
  rfl


@[simp]
theorem one_im : (1 : ℂ).im = 0 :=
  rfl


@[simp, norm_cast]
theorem ofReal_one : ((1 : ℝ) : ℂ) = 1 :=
  rfl


@[simp]
theorem ofReal_eq_one {z : ℝ} : (z : ℂ) = 1 ↔ z = 1 :=
  ofReal_inj


theorem ofReal_ne_one {z : ℝ} : (z : ℂ) ≠ 1 ↔ z ≠ 1 :=
  not_congr ofReal_eq_one


instance : Add ℂ :=
  ⟨fun z w => ⟨z.re + w.re, z.im + w.im⟩⟩


@[simp]
theorem add_re (z w : ℂ) : (z + w).re = z.re + w.re :=
  rfl


@[simp]
theorem add_im (z w : ℂ) : (z + w).im = z.im + w.im :=
  rfl

-- replaced by `re_ofNat`
-- replaced by `im_ofNat`


@[simp, norm_cast]
theorem ofReal_add (r s : ℝ) : ((r + s : ℝ) : ℂ) = r + s :=
                          /-
                            r s : Real
                            ⊢ And (Eq (↑(HAdd.hAdd r s)).re (HAdd.hAdd ↑r ↑s).re) (Eq (↑(HAdd.hAdd r s)).i …
                          -/
  Complex.ext_iff.2 <| by simp [ofReal]
                          /-
                            🎉 no goals
                          -/

-- replaced by `Complex.ofReal_ofNat`


instance : Neg ℂ :=
  ⟨fun z => ⟨-z.re, -z.im⟩⟩


@[simp]
theorem neg_re (z : ℂ) : (-z).re = -z.re :=
  rfl


@[simp]
theorem neg_im (z : ℂ) : (-z).im = -z.im :=
  rfl


@[simp, norm_cast]
theorem ofReal_neg (r : ℝ) : ((-r : ℝ) : ℂ) = -r :=
                          /-
                            r : Real
                            ⊢ And (Eq (↑(Neg.neg r)).re (Neg.neg ↑r).re) (Eq (↑(Neg.neg r)).im (Neg.neg ↑r …
                          -/
  Complex.ext_iff.2 <| by simp [ofReal]
                          /-
                            🎉 no goals
                          -/


instance : Sub ℂ :=
  ⟨fun z w => ⟨z.re - w.re, z.im - w.im⟩⟩


instance : Mul ℂ :=
  ⟨fun z w => ⟨z.re * w.re - z.im * w.im, z.re * w.im + z.im * w.re⟩⟩


@[simp]
theorem mul_re (z w : ℂ) : (z * w).re = z.re * w.re - z.im * w.im :=
  rfl


@[simp]
theorem mul_im (z w : ℂ) : (z * w).im = z.re * w.im + z.im * w.re :=
  rfl


@[simp, norm_cast]
theorem ofReal_mul (r s : ℝ) : ((r * s : ℝ) : ℂ) = r * s :=
                          /-
                            r s : Real
                            ⊢ And (Eq (↑(HMul.hMul r s)).re (HMul.hMul ↑r ↑s).re) (Eq (↑(HMul.hMul r s)).i …
                          -/
  Complex.ext_iff.2 <| by simp [ofReal]
                          /-
                            🎉 no goals
                          -/


                                                                    /-
                                                                      r : Real
                                                                      z : Complex
                                                                      ⊢ Eq (HMul.hMul (↑r) z).re (HMul.hMul r z.re)
                                                                    -/
theorem re_ofReal_mul (r : ℝ) (z : ℂ) : (r * z).re = r * z.re := by simp [ofReal]
                                                                    /-
                                                                      🎉 no goals
                                                                    -/


                                                                    /-
                                                                      r : Real
                                                                      z : Complex
                                                                      ⊢ Eq (HMul.hMul (↑r) z).im (HMul.hMul r z.im)
                                                                    -/
theorem im_ofReal_mul (r : ℝ) (z : ℂ) : (r * z).im = r * z.im := by simp [ofReal]
                                                                    /-
                                                                      🎉 no goals
                                                                    -/


                                                                   /-
                                                                     z : Complex
                                                                     r : Real
                                                                     ⊢ Eq (HMul.hMul z ↑r).re (HMul.hMul z.re r)
                                                                   -/
lemma re_mul_ofReal (z : ℂ) (r : ℝ) : (z * r).re = z.re *  r := by simp [ofReal]
                                                                   /-
                                                                     🎉 no goals
                                                                   -/

                                                                   /-
                                                                     z : Complex
                                                                     r : Real
                                                                     ⊢ Eq (HMul.hMul z ↑r).im (HMul.hMul z.im r)
                                                                   -/
lemma im_mul_ofReal (z : ℂ) (r : ℝ) : (z * r).im = z.im *  r := by simp [ofReal]
                                                                   /-
                                                                     🎉 no goals
                                                                   -/


theorem ofReal_mul' (r : ℝ) (z : ℂ) : ↑r * z = ⟨r * z.re, r * z.im⟩ :=
  ext (re_ofReal_mul _ _) (im_ofReal_mul _ _)


/-- The imaginary unit. -/
def I : ℂ :=
  ⟨0, 1⟩


@[simp]
theorem I_re : I.re = 0 :=
  rfl


@[simp]
theorem I_im : I.im = 1 :=
  rfl


@[simp]
theorem I_mul_I : I * I = -1 :=
                          /-
                            ⊢ And (Eq (HMul.hMul Complex.I Complex.I).re (-1).re) (Eq (HMul.hMul Complex.I …
                          -/
  Complex.ext_iff.2 <| by simp
                          /-
                            🎉 no goals
                          -/


theorem I_mul (z : ℂ) : I * z = ⟨-z.im, z.re⟩ :=
                          /-
                            z : Complex
                            ⊢ And (Eq (HMul.hMul Complex.I z).re { re := Neg.neg z.im, im := z.re }.re) (E …
                          -/
  Complex.ext_iff.2 <| by simp
                          /-
                            🎉 no goals
                          -/


@[simp] lemma I_ne_zero : (I : ℂ) ≠ 0 := mt (congr_arg im) zero_ne_one.symm


theorem mk_eq_add_mul_I (a b : ℝ) : Complex.mk a b = a + b * I :=
                          /-
                            a b : Real
                            ⊢ And (Eq { re := a, im := b }.re (HAdd.hAdd (↑a) (HMul.hMul (↑b) Complex.I)). …
                          -/
  Complex.ext_iff.2 <| by simp [ofReal]
                          /-
                            🎉 no goals
                          -/


@[simp]
theorem re_add_im (z : ℂ) : (z.re : ℂ) + z.im * I = z :=
                          /-
                            z : Complex
                            ⊢ And (Eq (HAdd.hAdd (↑z.re) (HMul.hMul (↑z.im) Complex.I)).re z.re) (Eq (HAdd …
                          -/
  Complex.ext_iff.2 <| by simp [ofReal]
                          /-
                            🎉 no goals
                          -/


                                                    /-
                                                      z : Complex
                                                      ⊢ Eq (HMul.hMul z Complex.I).re (Neg.neg z.im)
                                                    -/
theorem mul_I_re (z : ℂ) : (z * I).re = -z.im := by simp
                                                    /-
                                                      🎉 no goals
                                                    -/


                                                   /-
                                                     z : Complex
                                                     ⊢ Eq (HMul.hMul z Complex.I).im z.re
                                                   -/
theorem mul_I_im (z : ℂ) : (z * I).im = z.re := by simp
                                                   /-
                                                     🎉 no goals
                                                   -/


                                                    /-
                                                      z : Complex
                                                      ⊢ Eq (HMul.hMul Complex.I z).re (Neg.neg z.im)
                                                    -/
theorem I_mul_re (z : ℂ) : (I * z).re = -z.im := by simp
                                                    /-
                                                      🎉 no goals
                                                    -/


                                                   /-
                                                     z : Complex
                                                     ⊢ Eq (HMul.hMul Complex.I z).im z.re
                                                   -/
theorem I_mul_im (z : ℂ) : (I * z).im = z.re := by simp
                                                   /-
                                                     🎉 no goals
                                                   -/


@[simp]
theorem equivRealProd_symm_apply (p : ℝ × ℝ) : equivRealProd.symm p = p.1 + p.2 * I := by
  /-
    p : Prod Real Real
    ⊢ Eq (Complex.equivRealProd.symm p) (HAdd.hAdd (↑p.1) (HMul.hMul (↑p.2) Comple …
  -/
          /-
            🎉 no goals
          -/
  ext <;> simp [Complex.equivRealProd, ofReal]
          /-
            🎉 no goals
          -/


/-- The natural `AddEquiv` from `ℂ` to `ℝ × ℝ`. -/
@[simps! (config := { simpRhs := true }) apply symm_apply_re symm_apply_im]
def equivRealProdAddHom : ℂ ≃+ ℝ × ℝ :=
                                      /-
                                        ⊢ ∀ (x y : Complex), Eq (__src✝.toFun (HAdd.hAdd x y)) (HAdd.hAdd (__src✝.toFu …
                                      -/
  { equivRealProd with map_add' := by simp }
                                      /-
                                        🎉 no goals
                                      -/


theorem equivRealProdAddHom_symm_apply (p : ℝ × ℝ) :
    equivRealProdAddHom.symm p = p.1 + p.2 * I := equivRealProd_symm_apply p


instance : Nontrivial ℂ :=
  domain_nontrivial re rfl rfl

-- Porting note: moved from `Module/Data/Complex/Basic.lean`

/-- Scalar multiplication by `R` on `ℝ` extends to `ℂ`. This is used here and in
`Matlib.Data.Complex.Module` to transfer instances from `ℝ` to `ℂ`, but is not
needed outside, so we make it scoped. -/
scoped instance instSMulRealComplex {R : Type*} [SMul R ℝ] : SMul R ℂ where
  smul r x := ⟨r • x.re - 0 * x.im, r • x.im + 0 * x.re⟩


                                                              /-
                                                                R : Type u_1
                                                                inst✝ : SMul R Real
                                                                r : R
                                                                z : Complex
                                                                ⊢ Eq (HSMul.hSMul r z).re (HSMul.hSMul r z.re)
                                                              -/
theorem smul_re (r : R) (z : ℂ) : (r • z).re = r • z.re := by simp [(· • ·), SMul.smul]
                                                              /-
                                                                🎉 no goals
                                                              -/


                                                              /-
                                                                R : Type u_1
                                                                inst✝ : SMul R Real
                                                                r : R
                                                                z : Complex
                                                                ⊢ Eq (HSMul.hSMul r z).im (HSMul.hSMul r z.im)
                                                              -/
theorem smul_im (r : R) (z : ℂ) : (r • z).im = r • z.im := by simp [(· • ·), SMul.smul]
                                                              /-
                                                                🎉 no goals
                                                              -/


@[simp]
theorem real_smul {x : ℝ} {z : ℂ} : x • z = x * z :=
  rfl


instance addCommGroup : AddCommGroup ℂ :=
  { zero := (0 : ℂ)
    add := (· + ·)
    neg := Neg.neg
    sub := Sub.sub
    nsmul := fun n z => n • z
    zsmul := fun n z => n • z
                      /-
                        ⊢ ∀ (a : Complex), Eq ((fun n z => HSMul.hSMul n z) 0 a) 0
                      -/
                     /-
                       ⊢ ∀ (x : Complex), Eq ((fun n z => HSMul.hSMul n z) 0 x) 0
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
    zsmul_zero' := by intros; ext <;> simp [smul_re, smul_im]
      /-
        ⊢ ∀ (n : Nat) (x : Complex), Eq ((fun n z => HSMul.hSMul n z) (HAdd.hAdd n 1)  …
      -/
                                      /-
                                        🎉 no goals
                                      -/
    nsmul_zero := by intros; ext <;> simp [smul_re, smul_im]
                    /-
                      ⊢ ∀ (a b c : Complex), Eq (HAdd.hAdd (HAdd.hAdd a b) c) (HAdd.hAdd a (HAdd.hAd …
                    -/
                                    /-
                                      🎉 no goals
                                    -/
    nsmul_succ := by
                                    /-
                                      🎉 no goals
                                    -/
                   /-
                     ⊢ ∀ (a : Complex), Eq (HAdd.hAdd 0 a) a
                   -/
                                   /-
                                     🎉 no goals
                                   -/
      intros; ext <;> simp [AddMonoid.nsmul_succ, add_mul, add_comm,
                                   /-
                                     🎉 no goals
                                   -/
                   /-
                     ⊢ ∀ (a : Complex), Eq (HAdd.hAdd a 0) a
                   -/
                                   /-
                                     🎉 no goals
                                   -/
        smul_re, smul_im]
                                   /-
                                     🎉 no goals
                                   -/
    zsmul_succ' := by
      /-
        ⊢ ∀ (n : Nat) (a : Complex), Eq ((fun n z => HSMul.hSMul n z) (↑n.succ) a) (HA …
      -/
                      /-
                        🎉 no goals
                      -/
      intros; ext <;> simp [add_mul, smul_re, smul_im]
                      /-
                        🎉 no goals
                      -/
    zsmul_neg' := by
      /-
        ⊢ ∀ (n : Nat) (a : Complex), Eq ((fun n z => HSMul.hSMul n z) (Int.negSucc n)  …
      -/
                      /-
                        🎉 no goals
                      -/
      intros; ext <;> simp [zsmul_neg', add_mul, smul_re, smul_im]
                      /-
                        🎉 no goals
                      -/
    add_assoc := by intros; ext <;> simp [add_assoc]
    zero_add := by intros; ext <;> simp
    add_zero := by intros; ext <;> simp
                   /-
                     ⊢ ∀ (a b : Complex), Eq (HAdd.hAdd a b) (HAdd.hAdd b a)
                   -/
                         /-
                           ⊢ ∀ (a : Complex), Eq (HAdd.hAdd (Neg.neg a) a) 0
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
    add_comm := by intros; ext <;> simp [add_comm]
                                   /-
                                     🎉 no goals
                                   -/
    neg_add_cancel := by intros; ext <;> simp }



instance addGroupWithOne : AddGroupWithOne ℂ :=
  { Complex.addCommGroup with
    natCast := fun n => ⟨n, 0⟩
    natCast_zero := by
      /-
        ⊢ Eq (NatCast.natCast 0) 0
      -/
              /-
                🎉 no goals
              -/
      ext <;> simp [Nat.cast, AddMonoidWithOne.natCast_zero]
              /-
                🎉 no goals
              -/
                                /-
                                  x✝ : Nat
                                  ⊢ Eq (NatCast.natCast (HAdd.hAdd x✝ 1)) (HAdd.hAdd (NatCast.natCast x✝) 1)
                                -/
                                        /-
                                          🎉 no goals
                                        -/
    natCast_succ := fun _ => by ext <;> simp [Nat.cast, AddMonoidWithOne.natCast_succ]
                                        /-
                                          🎉 no goals
                                        -/
    intCast := fun n => ⟨n, 0⟩
                                 /-
                                   x✝ : Nat
                                   ⊢ Eq (IntCast.intCast ↑x✝) ↑x✝
                                 -/
                                         /-
                                           🎉 no goals
                                         -/
    intCast_ofNat := fun _ => by ext <;> rfl
                                         /-
                                           🎉 no goals
                                         -/
    intCast_negSucc := fun n => by
      /-
        n : Nat
        ⊢ Eq (IntCast.intCast (Int.negSucc n)) (Neg.neg ↑(HAdd.hAdd n 1))
      -/
      ext
        /-
          case a
          n : Nat
          ⊢ Eq (IntCast.intCast (Int.negSucc n)).re (Neg.neg ↑(HAdd.hAdd n 1)).re
        -/
      · simp [AddGroupWithOne.intCast_negSucc]
        /-
          case a
          n : Nat
          ⊢ Eq (HAdd.hAdd (-1) (Neg.neg ↑n)) (Neg.neg (↑(HAdd.hAdd n 1)).re)
        -/
        show -(1 : ℝ) + (-n) = -(↑(n + 1))
        /-
          case a
          n : Nat
          ⊢ Eq (HAdd.hAdd (-1) (Neg.neg ↑n)) (Neg.neg ↑(HAdd.hAdd n 1))
        -/
        simp [Nat.cast_add, add_comm]
        /-
          🎉 no goals
        -/
        /-
          case a
          n : Nat
          ⊢ Eq (IntCast.intCast (Int.negSucc n)).im (Neg.neg ↑(HAdd.hAdd n 1)).im
        -/
      · simp [AddGroupWithOne.intCast_negSucc]
        /-
          case a
          n : Nat
          ⊢ Eq (↑(HAdd.hAdd n 1)).im 0
        -/
        show im ⟨n, 0⟩ = 0
        /-
          case a
          n : Nat
          ⊢ Eq { re := ↑n, im := 0 }.im 0
        -/
        rfl
        /-
          🎉 no goals
        -/
    one := 1 }

-- Porting note: proof needed modifications and rewritten fields

instance commRing : CommRing ℂ :=
  { addGroupWithOne with
    mul := (· * ·)
    npow := @npowRec _ ⟨(1 : ℂ)⟩ ⟨(· * ·)⟩
                   /-
                     ⊢ ∀ (a b : Complex), Eq (HAdd.hAdd a b) (HAdd.hAdd b a)
                   -/
                                   /-
                                     🎉 no goals
                                   -/
    add_comm := by intros; ext <;> simp [add_comm]
                                   /-
                                     🎉 no goals
                                   -/
    left_distrib := by
      /-
        ⊢ ∀ (a b c : Complex), Eq (HMul.hMul a (HAdd.hAdd b c)) (HAdd.hAdd (HMul.hMul  …
      -/
                                                /-
                                                  🎉 no goals
                                                -/
      intros; ext <;> simp [mul_re, mul_im] <;> ring
                                                /-
                                                  🎉 no goals
                                                -/
    right_distrib := by
      /-
        ⊢ ∀ (a b c : Complex), Eq (HMul.hMul (HAdd.hAdd a b) c) (HAdd.hAdd (HMul.hMul  …
      -/
                                                /-
                                                  🎉 no goals
                                                -/
      intros; ext <;> simp [mul_re, mul_im] <;> ring
                                                /-
                                                  🎉 no goals
                                                -/
                   /-
                     ⊢ ∀ (a : Complex), Eq (HMul.hMul 0 a) 0
                   -/
                                   /-
                                     🎉 no goals
                                   -/
    zero_mul := by intros; ext <;> simp [zero_mul]
                                   /-
                                     🎉 no goals
                                   -/
                   /-
                     ⊢ ∀ (a : Complex), Eq (HMul.hMul a 0) 0
                   -/
                                   /-
                                     🎉 no goals
                                   -/
    mul_zero := by intros; ext <;> simp [mul_zero]
                                   /-
                                     🎉 no goals
                                   -/
                    /-
                      ⊢ ∀ (a b c : Complex), Eq (HMul.hMul (HMul.hMul a b) c) (HMul.hMul a (HMul.hMu …
                    -/
                                                         /-
                                                           🎉 no goals
                                                         -/
    mul_assoc := by intros; ext <;> simp [mul_assoc] <;> ring
                                                         /-
                                                           🎉 no goals
                                                         -/
                  /-
                    ⊢ ∀ (a : Complex), Eq (HMul.hMul 1 a) a
                  -/
                                  /-
                                    🎉 no goals
                                  -/
    one_mul := by intros; ext <;> simp [one_mul]
                                  /-
                                    🎉 no goals
                                  -/
                  /-
                    ⊢ ∀ (a : Complex), Eq (HMul.hMul a 1) a
                  -/
                                  /-
                                    🎉 no goals
                                  -/
    mul_one := by intros; ext <;> simp [mul_one]
                                  /-
                                    🎉 no goals
                                  -/
                   /-
                     ⊢ ∀ (a b : Complex), Eq (HMul.hMul a b) (HMul.hMul b a)
                   -/
                                   /-
                                     🎉 no goals
                                   -/
    mul_comm := by intros; ext <;> simp [mul_comm]; ring }
                                                    /-
                                                      🎉 no goals
                                                    -/


/-- This shortcut instance ensures we do not find `Ring` via the noncomputable `Complex.field`
instance. -/
                        /-
                          ⊢ Ring Complex
                        -/
instance : Ring ℂ := by infer_instance
                        /-
                          🎉 no goals
                        -/


/-- This shortcut instance ensures we do not find `CommSemiring` via the noncomputable
`Complex.field` instance. -/
instance : CommSemiring ℂ :=
  inferInstance

-- Porting note: added due to changes in typeclass search order

/-- This shortcut instance ensures we do not find `Semiring` via the noncomputable
`Complex.field` instance. -/
instance : Semiring ℂ :=
  inferInstance


/-- The "real part" map, considered as an additive group homomorphism. -/
def reAddGroupHom : ℂ →+ ℝ where
  toFun := re
  map_zero' := zero_re
  map_add' := add_re


@[simp]
theorem coe_reAddGroupHom : (reAddGroupHom : ℂ → ℝ) = re :=
  rfl


/-- The "imaginary part" map, considered as an additive group homomorphism. -/
def imAddGroupHom : ℂ →+ ℝ where
  toFun := im
  map_zero' := zero_im
  map_add' := add_im


@[simp]
theorem coe_imAddGroupHom : (imAddGroupHom : ℂ → ℝ) = im :=
  rfl


noncomputable instance instNNRatCast : NNRatCast ℂ where nnratCast q := ofReal q

noncomputable instance instRatCast : RatCast ℂ where ratCast q := ofReal q


@[simp, norm_cast] lemma ofReal_ofNat (n : ℕ) [n.AtLeastTwo] : ofReal ofNat(n) = ofNat(n) := rfl

@[simp, norm_cast] lemma ofReal_natCast (n : ℕ) : ofReal n = n := rfl

@[simp, norm_cast] lemma ofReal_intCast (n : ℤ) : ofReal n = n := rfl

@[simp, norm_cast] lemma ofReal_nnratCast (q : ℚ≥0) : ofReal q = q := rfl

@[simp, norm_cast] lemma ofReal_ratCast (q : ℚ) : ofReal q = q := rfl


@[deprecated (since := "2024-04-17")]
alias ofReal_rat_cast := ofReal_ratCast


@[simp]
lemma re_ofNat (n : ℕ) [n.AtLeastTwo] : (ofNat(n) : ℂ).re = ofNat(n) := rfl

@[simp] lemma im_ofNat (n : ℕ) [n.AtLeastTwo] : (ofNat(n) : ℂ).im = 0 := rfl

@[simp, norm_cast] lemma natCast_re (n : ℕ) : (n : ℂ).re = n := rfl

@[simp, norm_cast] lemma natCast_im (n : ℕ) : (n : ℂ).im = 0 := rfl

@[simp, norm_cast] lemma intCast_re (n : ℤ) : (n : ℂ).re = n := rfl

@[simp, norm_cast] lemma intCast_im (n : ℤ) : (n : ℂ).im = 0 := rfl

@[simp, norm_cast] lemma re_nnratCast (q : ℚ≥0) : (q : ℂ).re = q := rfl

@[simp, norm_cast] lemma im_nnratCast (q : ℚ≥0) : (q : ℂ).im = 0 := rfl

@[simp, norm_cast] lemma ratCast_re (q : ℚ) : (q : ℂ).re = q := rfl

@[simp, norm_cast] lemma ratCast_im (q : ℚ) : (q : ℂ).im = 0 := rfl


lemma re_nsmul (n : ℕ) (z : ℂ) : (n • z).re = n • z.re := smul_re ..

lemma im_nsmul (n : ℕ) (z : ℂ) : (n • z).im = n • z.im := smul_im ..

lemma re_zsmul (n : ℤ) (z : ℂ) : (n • z).re = n • z.re := smul_re ..

lemma im_zsmul (n : ℤ) (z : ℂ) : (n • z).im = n • z.im := smul_im ..

@[simp] lemma re_nnqsmul (q : ℚ≥0) (z : ℂ) : (q • z).re = q • z.re := smul_re ..

@[simp] lemma im_nnqsmul (q : ℚ≥0) (z : ℂ) : (q • z).im = q • z.im := smul_im ..

@[simp] lemma re_qsmul (q : ℚ) (z : ℂ) : (q • z).re = q • z.re := smul_re ..

@[simp] lemma im_qsmul (q : ℚ) (z : ℂ) : (q • z).im = q • z.im := smul_im ..


@[deprecated (since := "2024-04-17")]
alias rat_cast_im := ratCast_im


                                                                               /-
                                                                                 n : Nat
                                                                                 r : Real
                                                                                 ⊢ Eq (↑(HSMul.hSMul n r)) (HSMul.hSMul n ↑r)
                                                                               -/
@[norm_cast] lemma ofReal_nsmul (n : ℕ) (r : ℝ) : ↑(n • r) = n • (r : ℂ) := by simp
                                                                               /-
                                                                                 🎉 no goals
                                                                               -/

                                                                               /-
                                                                                 n : Int
                                                                                 r : Real
                                                                                 ⊢ Eq (↑(HSMul.hSMul n r)) (HSMul.hSMul n ↑r)
                                                                               -/
@[norm_cast] lemma ofReal_zsmul (n : ℤ) (r : ℝ) : ↑(n • r) = n • (r : ℂ) := by simp
                                                                               /-
                                                                                 🎉 no goals
                                                                               -/


/-- This defines the complex conjugate as the `star` operation of the `StarRing ℂ`. It
is recommended to use the ring endomorphism version `starRingEnd`, available under the
notation `conj` in the locale `ComplexConjugate`. -/
instance : StarRing ℂ where
  star z := ⟨z.re, -z.im⟩
                          /-
                            x : Complex
                            ⊢ Eq (Star.star (Star.star x)) x
                          -/
  star_involutive x := by simp only [eta, neg_neg]
                          /-
                            🎉 no goals
                          -/
                     /-
                       a b : Complex
                       ⊢ Eq (Star.star (HMul.hMul a b)) (HMul.hMul (Star.star b) (Star.star a))
                     -/
                                                 /-
                                                   🎉 no goals
                                                 -/
  star_mul a b := by ext <;> simp [add_comm] <;> ring
                                                 /-
                                                   🎉 no goals
                                                 -/
                     /-
                       a b : Complex
                       ⊢ Eq (Star.star (HAdd.hAdd a b)) (HAdd.hAdd (Star.star a) (Star.star b))
                     -/
                             /-
                               🎉 no goals
                             -/
  star_add a b := by ext <;> simp [add_comm]
                             /-
                               🎉 no goals
                             -/


@[simp]
theorem conj_re (z : ℂ) : (conj z).re = z.re :=
  rfl


@[simp]
theorem conj_im (z : ℂ) : (conj z).im = -z.im :=
  rfl


@[simp]
theorem conj_ofReal (r : ℝ) : conj (r : ℂ) = r :=
                          /-
                            r : Real
                            ⊢ And (Eq ((starRingEnd Complex) ↑r).re (↑r).re) (Eq ((starRingEnd Complex) ↑r …
                          -/
  Complex.ext_iff.2 <| by simp [star]
                          /-
                            🎉 no goals
                          -/


@[simp]
theorem conj_I : conj I = -I :=
                          /-
                            ⊢ And (Eq ((starRingEnd Complex) Complex.I).re (Neg.neg Complex.I).re) (Eq ((s …
                          -/
  Complex.ext_iff.2 <| by simp
                          /-
                            🎉 no goals
                          -/


theorem conj_natCast (n : ℕ) : conj (n : ℂ) = n := map_natCast _ _


@[deprecated (since := "2024-04-17")]
alias conj_nat_cast := conj_natCast


theorem conj_ofNat (n : ℕ) [n.AtLeastTwo] : conj (ofNat(n) : ℂ) = ofNat(n) :=
  map_ofNat _ _

-- @[simp]
/- Porting note (https://github.com/leanprover-community/mathlib4/issues/11119): `simp` attribute removed as the result could be proved
by `simp only [@map_neg, Complex.conj_i, @neg_neg]`
-/

theorem conj_neg_I : conj (-I) = I :=
                          /-
                            ⊢ And (Eq ((starRingEnd Complex) (Neg.neg Complex.I)).re Complex.I.re) (Eq ((s …
                          -/
  Complex.ext_iff.2 <| by simp
                          /-
                            🎉 no goals
                          -/


theorem conj_eq_iff_real {z : ℂ} : conj z = z ↔ ∃ r : ℝ, z = r :=
  ⟨fun h => ⟨z.re, ext rfl <| eq_zero_of_neg_eq (congr_arg im h)⟩, fun ⟨h, e⟩ => by
    /-
      z : Complex
      x✝ : Exists fun r => Eq z ↑r
      h : Real
      e : Eq z ↑h
      ⊢ Eq ((starRingEnd Complex) z) z
    -/
    rw [e, conj_ofReal]⟩
    /-
      🎉 no goals
    -/


theorem conj_eq_iff_re {z : ℂ} : conj z = z ↔ (z.re : ℂ) = z :=
                             /-
                               z : Complex
                               ⊢ (Exists fun r => Eq z ↑r) → Eq (↑z.re) z
                             -/
  conj_eq_iff_real.trans ⟨by rintro ⟨r, rfl⟩; simp [ofReal], fun h => ⟨_, h.symm⟩⟩
                                              /-
                                                🎉 no goals
                                              -/


theorem conj_eq_iff_im {z : ℂ} : conj z = z ↔ z.im = 0 :=
  ⟨fun h => add_self_eq_zero.mp (neg_eq_iff_add_eq_zero.mp (congr_arg im h)), fun h =>
    ext rfl (neg_eq_iff_add_eq_zero.mpr (add_self_eq_zero.mpr h))⟩

-- `simpNF` complains about this being provable by `RCLike.star_def` even
-- though it's not imported by this file.
-- Porting note: linter `simpNF` not found

@[simp]
theorem star_def : (Star.star : ℂ → ℂ) = conj :=
  rfl


/-- The norm squared function. -/
@[pp_nodot]
def normSq : ℂ →*₀ ℝ where
  toFun z := z.re * z.re + z.im * z.im
                  /-
                    ⊢ Eq ((fun z => HAdd.hAdd (HMul.hMul z.re z.re) (HMul.hMul z.im z.im)) 0) 0
                  -/
  map_zero' := by simp
                  /-
                    🎉 no goals
                  -/
                 /-
                   ⊢ Eq ({ toFun := fun z => HAdd.hAdd (HMul.hMul z.re z.re) (HMul.hMul z.im z.im …
                 -/
  map_one' := by simp
                 /-
                   🎉 no goals
                 -/
  map_mul' z w := by
    /-
      z w : Complex
      ⊢ Eq ({ toFun := fun z => HAdd.hAdd (HMul.hMul z.re z.re) (HMul.hMul z.im z.im …
    -/
    dsimp
    /-
      z w : Complex
      ⊢ Eq (HAdd.hAdd (HMul.hMul (HSub.hSub (HMul.hMul z.re w.re) (HMul.hMul z.im w. …
    -/
    ring
    /-
      🎉 no goals
    -/


theorem normSq_apply (z : ℂ) : normSq z = z.re * z.re + z.im * z.im :=
  rfl


@[simp]
theorem normSq_ofReal (r : ℝ) : normSq r = r * r := by
  /-
    r : Real
    ⊢ Eq (Complex.normSq ↑r) (HMul.hMul r r)
  -/
  simp [normSq, ofReal]
  /-
    🎉 no goals
  -/


@[simp]
theorem normSq_natCast (n : ℕ) : normSq n = n * n := normSq_ofReal _


@[deprecated (since := "2024-04-17")]
alias normSq_nat_cast := normSq_natCast


@[simp]
theorem normSq_intCast (z : ℤ) : normSq z = z * z := normSq_ofReal _


@[deprecated (since := "2024-04-17")]
alias normSq_int_cast := normSq_intCast


@[simp]
theorem normSq_ratCast (q : ℚ) : normSq q = q * q := normSq_ofReal _


@[deprecated (since := "2024-04-17")]
alias normSq_rat_cast := normSq_ratCast


@[simp]
theorem normSq_ofNat (n : ℕ) [n.AtLeastTwo] :
    normSq (ofNat(n) : ℂ) = ofNat(n) * ofNat(n) :=
  normSq_natCast _


@[simp]
theorem normSq_mk (x y : ℝ) : normSq ⟨x, y⟩ = x * x + y * y :=
  rfl


theorem normSq_add_mul_I (x y : ℝ) : normSq (x + y * I) = x ^ 2 + y ^ 2 := by
  /-
    x y : Real
    ⊢ Eq (Complex.normSq (HAdd.hAdd (↑x) (HMul.hMul (↑y) Complex.I))) (HAdd.hAdd ( …
  -/
  rw [← mk_eq_add_mul_I, normSq_mk, sq, sq]
  /-
    🎉 no goals
  -/


theorem normSq_eq_conj_mul_self {z : ℂ} : (normSq z : ℂ) = conj z * z := by
  /-
    z : Complex
    ⊢ Eq (↑(Complex.normSq z)) (HMul.hMul ((starRingEnd Complex) z) z)
  -/
          /-
            🎉 no goals
          -/
  ext <;> simp [normSq, mul_comm, ofReal]
          /-
            🎉 no goals
          -/

-- @[simp]
/- Porting note (https://github.com/leanprover-community/mathlib4/issues/11119): `simp` attribute removed as linter reports this can be proved
by `simp only [@map_zero]` -/

theorem normSq_zero : normSq 0 = 0 :=
  normSq.map_zero

-- @[simp]
/- Porting note (https://github.com/leanprover-community/mathlib4/issues/11119): `simp` attribute removed as linter reports this can be proved
by `simp only [@map_one]` -/

theorem normSq_one : normSq 1 = 1 :=
  normSq.map_one


@[simp]
                                      /-
                                        ⊢ Eq (Complex.normSq Complex.I) 1
                                      -/
theorem normSq_I : normSq I = 1 := by simp [normSq]
                                      /-
                                        🎉 no goals
                                      -/


theorem normSq_nonneg (z : ℂ) : 0 ≤ normSq z :=
  add_nonneg (mul_self_nonneg _) (mul_self_nonneg _)


theorem normSq_eq_zero {z : ℂ} : normSq z = 0 ↔ z = 0 :=
  ⟨fun h =>
    ext (eq_zero_of_mul_self_add_mul_self_eq_zero h)
      (eq_zero_of_mul_self_add_mul_self_eq_zero <| (add_comm _ _).trans h),
    fun h => h.symm ▸ normSq_zero⟩


@[simp]
theorem normSq_pos {z : ℂ} : 0 < normSq z ↔ z ≠ 0 :=
  (normSq_nonneg z).lt_iff_ne.trans <| not_congr (eq_comm.trans normSq_eq_zero)


@[simp]
                                                          /-
                                                            z : Complex
                                                            ⊢ Eq (Complex.normSq (Neg.neg z)) (Complex.normSq z)
                                                          -/
theorem normSq_neg (z : ℂ) : normSq (-z) = normSq z := by simp [normSq]
                                                          /-
                                                            🎉 no goals
                                                          -/


@[simp]
                                                               /-
                                                                 z : Complex
                                                                 ⊢ Eq (Complex.normSq ((starRingEnd Complex) z)) (Complex.normSq z)
                                                               -/
theorem normSq_conj (z : ℂ) : normSq (conj z) = normSq z := by simp [normSq]
                                                               /-
                                                                 🎉 no goals
                                                               -/


theorem normSq_mul (z w : ℂ) : normSq (z * w) = normSq z * normSq w :=
  normSq.map_mul z w


theorem normSq_add (z w : ℂ) : normSq (z + w) = normSq z + normSq w + 2 * (z * conj w).re := by
  /-
    z w : Complex
    ⊢ Eq (Complex.normSq (HAdd.hAdd z w)) (HAdd.hAdd (HAdd.hAdd (Complex.normSq z) …
  -/
  dsimp [normSq]; ring
                  /-
                    🎉 no goals
                  -/


theorem re_sq_le_normSq (z : ℂ) : z.re * z.re ≤ normSq z :=
  le_add_of_nonneg_right (mul_self_nonneg _)


theorem im_sq_le_normSq (z : ℂ) : z.im * z.im ≤ normSq z :=
  le_add_of_nonneg_left (mul_self_nonneg _)


theorem mul_conj (z : ℂ) : z * conj z = normSq z :=
                          /-
                            z : Complex
                            ⊢ And (Eq (HMul.hMul z ((starRingEnd Complex) z)).re (↑(Complex.normSq z)).re) …
                          -/
  Complex.ext_iff.2 <| by simp [normSq, mul_comm, sub_eq_neg_add, add_comm, ofReal]
                          /-
                            🎉 no goals
                          -/


theorem add_conj (z : ℂ) : z + conj z = (2 * z.re : ℝ) :=
                          /-
                            z : Complex
                            ⊢ And (Eq (HAdd.hAdd z ((starRingEnd Complex) z)).re (↑(HMul.hMul 2 z.re)).re) …
                          -/
  Complex.ext_iff.2 <| by simp [two_mul, ofReal]
                          /-
                            🎉 no goals
                          -/


/-- The coercion `ℝ → ℂ` as a `RingHom`. -/
def ofRealHom : ℝ →+* ℂ where
  toFun x := (x : ℂ)
  map_one' := ofReal_one
  map_zero' := ofReal_zero
  map_mul' := ofReal_mul
  map_add' := ofReal_add


@[simp] lemma ofRealHom_eq_coe (r : ℝ) : ofRealHom r = r := rfl


@[simp] lemma ofReal_comp_add (f g : α → ℝ) : ofReal ∘ (f + g) = ofReal ∘ f + ofReal ∘ g :=
  map_comp_add ofRealHom ..


@[simp] lemma ofReal_comp_sub (f g : α → ℝ) : ofReal ∘ (f - g) = ofReal ∘ f - ofReal ∘ g :=
  map_comp_sub ofRealHom ..


@[simp] lemma ofReal_comp_neg (f : α → ℝ) : ofReal ∘ (-f) = -(ofReal ∘ f) :=
  map_comp_neg ofRealHom _


lemma ofReal_comp_nsmul (n : ℕ) (f : α → ℝ) : ofReal ∘ (n • f) = n • (ofReal ∘ f) :=
  map_comp_nsmul ofRealHom ..


lemma ofReal_comp_zsmul (n : ℤ) (f : α → ℝ) : ofReal ∘ (n • f) = n • (ofReal ∘ f) :=
  map_comp_zsmul ofRealHom ..


@[simp] lemma ofReal_comp_mul (f g : α → ℝ) : ofReal ∘ (f * g) = ofReal ∘ f * ofReal ∘ g :=
  map_comp_mul ofRealHom ..


@[simp] lemma ofReal_comp_pow (f : α → ℝ) (n : ℕ) : ofReal ∘ (f ^ n) = (ofReal ∘ f) ^ n :=
  map_comp_pow ofRealHom ..


@[simp]
                                /-
                                  ⊢ Eq (HPow.hPow Complex.I 2) (-1)
                                -/
theorem I_sq : I ^ 2 = -1 := by rw [sq, I_mul_I]
                                /-
                                  🎉 no goals
                                -/


@[simp]
                                     /-
                                       ⊢ Eq (HPow.hPow Complex.I 4) 1
                                     -/
theorem I_pow_four : I ^ 4 = 1 := by rw [(by norm_num : 4 = 2 * 2), pow_mul, I_sq, neg_one_sq]
                                     /-
                                       🎉 no goals
                                     -/


@[simp]
theorem sub_re (z w : ℂ) : (z - w).re = z.re - w.re :=
  rfl


@[simp]
theorem sub_im (z w : ℂ) : (z - w).im = z.im - w.im :=
  rfl


@[simp, norm_cast]
theorem ofReal_sub (r s : ℝ) : ((r - s : ℝ) : ℂ) = r - s :=
                          /-
                            r s : Real
                            ⊢ And (Eq (↑(HSub.hSub r s)).re (HSub.hSub ↑r ↑s).re) (Eq (↑(HSub.hSub r s)).i …
                          -/
  Complex.ext_iff.2 <| by simp [ofReal]
                          /-
                            🎉 no goals
                          -/


@[simp, norm_cast]
theorem ofReal_pow (r : ℝ) (n : ℕ) : ((r ^ n : ℝ) : ℂ) = (r : ℂ) ^ n := by
  /-
    r : Real
    n : Nat
    ⊢ Eq (↑(HPow.hPow r n)) (HPow.hPow (↑r) n)
  -/
                  /-
                    🎉 no goals
                  -/
  induction n <;> simp [*, ofReal_mul, pow_succ]
                  /-
                    🎉 no goals
                  -/


theorem sub_conj (z : ℂ) : z - conj z = (2 * z.im : ℝ) * I :=
                          /-
                            z : Complex
                            ⊢ And (Eq (HSub.hSub z ((starRingEnd Complex) z)).re (HMul.hMul (↑(HMul.hMul 2 …
                          -/
  Complex.ext_iff.2 <| by simp [two_mul, sub_eq_add_neg, ofReal]
                          /-
                            🎉 no goals
                          -/


theorem normSq_sub (z w : ℂ) : normSq (z - w) = normSq z + normSq w - 2 * (z * conj w).re := by
  /-
    z w : Complex
    ⊢ Eq (Complex.normSq (HSub.hSub z w)) (HSub.hSub (HAdd.hAdd (Complex.normSq z) …
  -/
  rw [sub_eq_add_neg, normSq_add]
  /-
    z w : Complex
    ⊢ Eq (HAdd.hAdd (HAdd.hAdd (Complex.normSq z) (Complex.normSq (Neg.neg w))) (H …
  -/
  simp only [RingHom.map_neg, mul_neg, neg_re, normSq_neg]
  /-
    z w : Complex
    ⊢ Eq (HAdd.hAdd (HAdd.hAdd (Complex.normSq z) (Complex.normSq w)) (Neg.neg (HM …
  -/
  ring
  /-
    🎉 no goals
  -/


noncomputable instance : Inv ℂ :=
  ⟨fun z => conj z * ((normSq z)⁻¹ : ℝ)⟩


theorem inv_def (z : ℂ) : z⁻¹ = conj z * ((normSq z)⁻¹ : ℝ) :=
  rfl


@[simp]
                                                        /-
                                                          z : Complex
                                                          ⊢ Eq (Inv.inv z).re (HDiv.hDiv z.re (Complex.normSq z))
                                                        -/
theorem inv_re (z : ℂ) : z⁻¹.re = z.re / normSq z := by simp [inv_def, division_def, ofReal]
                                                        /-
                                                          🎉 no goals
                                                        -/


@[simp]
                                                         /-
                                                           z : Complex
                                                           ⊢ Eq (Inv.inv z).im (HDiv.hDiv (Neg.neg z.im) (Complex.normSq z))
                                                         -/
theorem inv_im (z : ℂ) : z⁻¹.im = -z.im / normSq z := by simp [inv_def, division_def, ofReal]
                                                         /-
                                                           🎉 no goals
                                                         -/


@[simp, norm_cast]
theorem ofReal_inv (r : ℝ) : ((r⁻¹ : ℝ) : ℂ) = (r : ℂ)⁻¹ :=
                          /-
                            r : Real
                            ⊢ And (Eq (↑(Inv.inv r)).re (Inv.inv ↑r).re) (Eq (↑(Inv.inv r)).im (Inv.inv ↑r …
                          -/
  Complex.ext_iff.2 <| by simp [ofReal]
                          /-
                            🎉 no goals
                          -/


protected theorem inv_zero : (0⁻¹ : ℂ) = 0 := by
  /-
    ⊢ Eq (Inv.inv 0) 0
  -/
  rw [← ofReal_zero, ← ofReal_inv, inv_zero]
  /-
    🎉 no goals
  -/


protected theorem mul_inv_cancel {z : ℂ} (h : z ≠ 0) : z * z⁻¹ = 1 := by
  rw [inv_def, ← mul_assoc, mul_conj, ← ofReal_mul, mul_inv_cancel₀ (mt normSq_eq_zero.1 h),
    ofReal_one]


noncomputable instance instDivInvMonoid : DivInvMonoid ℂ where


lemma div_re (z w : ℂ) : (z / w).re = z.re * w.re / normSq w + z.im * w.im / normSq w := by
  /-
    z w : Complex
    ⊢ Eq (HDiv.hDiv z w).re (HAdd.hAdd (HDiv.hDiv (HMul.hMul z.re w.re) (Complex.n …
  -/
  simp [div_eq_mul_inv, mul_assoc, sub_eq_add_neg]
  /-
    🎉 no goals
  -/


lemma div_im (z w : ℂ) : (z / w).im = z.im * w.re / normSq w - z.re * w.im / normSq w := by
  /-
    z w : Complex
    ⊢ Eq (HDiv.hDiv z w).im (HSub.hSub (HDiv.hDiv (HMul.hMul z.im w.re) (Complex.n …
  -/
  simp [div_eq_mul_inv, mul_assoc, sub_eq_add_neg, add_comm]
  /-
    🎉 no goals
  -/


noncomputable instance instField : Field ℂ where
  mul_inv_cancel := @Complex.mul_inv_cancel
  inv_zero := Complex.inv_zero
  nnqsmul := (· • ·)
  qsmul := (· • ·)
                        /-
                          α : Type u_1
                          q : NNRat
                          ⊢ Eq (↑q) (HDiv.hDiv ↑q.num ↑q.den)
                        -/
                                /-
                                  🎉 no goals
                                -/
  nnratCast_def q := by ext <;> simp [NNRat.cast_def, div_re, div_im, mul_div_mul_comm]
                                /-
                                  🎉 no goals
                                -/
                      /-
                        α : Type u_1
                        q : Rat
                        ⊢ Eq (↑q) (HDiv.hDiv ↑q.num ↑q.den)
                      -/
                                             /-
                                               α : Type u_1
                                               n : NNRat
                                               z : Complex
                                               ⊢ And (Eq ((fun x1 x2 => HSMul.hSMul x1 x2) n z).re (HMul.hMul (↑n) z).re) (Eq …
                                             -/
                              /-
                                🎉 no goals
                              -/
                                             /-
                                               🎉 no goals
                                             -/
  ratCast_def q := by ext <;> simp [Rat.cast_def, div_re, div_im, mul_div_mul_comm]
                              /-
                                🎉 no goals
                              -/
  nnqsmul_def n z := Complex.ext_iff.2 <| by simp [NNRat.smul_def, smul_re, smul_im]
                                           /-
                                             α : Type u_1
                                             n : Rat
                                             z : Complex
                                             ⊢ And (Eq ((fun x1 x2 => HSMul.hSMul x1 x2) n z).re (HMul.hMul (↑n) z).re) (Eq …
                                           -/
  qsmul_def n z := Complex.ext_iff.2 <| by simp [Rat.smul_def, smul_re, smul_im]
                                           /-
                                             🎉 no goals
                                           -/


@[simp, norm_cast]
                                                                      /-
                                                                        q : NNRat
                                                                        r : Real
                                                                        ⊢ Eq (↑(HSMul.hSMul q r)) (HSMul.hSMul q ↑r)
                                                                      -/
lemma ofReal_nnqsmul (q : ℚ≥0) (r : ℝ) : ofReal (q • r) = q • r := by simp [NNRat.smul_def]
                                                                      /-
                                                                        🎉 no goals
                                                                      -/


@[simp, norm_cast]
                                                                  /-
                                                                    q : Rat
                                                                    r : Real
                                                                    ⊢ Eq (↑(HSMul.hSMul q r)) (HSMul.hSMul q ↑r)
                                                                  -/
lemma ofReal_qsmul (q : ℚ) (r : ℝ) : ofReal (q • r) = q • r := by simp [Rat.smul_def]
                                                                  /-
                                                                    🎉 no goals
                                                                  -/


theorem conj_inv (x : ℂ) : conj x⁻¹ = (conj x)⁻¹ :=
  star_inv₀ _


@[simp, norm_cast]
theorem ofReal_div (r s : ℝ) : ((r / s : ℝ) : ℂ) = r / s := map_div₀ ofRealHom r s


@[simp, norm_cast]
theorem ofReal_zpow (r : ℝ) (n : ℤ) : ((r ^ n : ℝ) : ℂ) = (r : ℂ) ^ n := map_zpow₀ ofRealHom r n


@[simp]
theorem div_I (z : ℂ) : z / I = -(z * I) :=
                                        /-
                                          z : Complex
                                          ⊢ Eq (HMul.hMul (Neg.neg (HMul.hMul z Complex.I)) Complex.I) z
                                        -/
  (div_eq_iff_mul_eq I_ne_zero).2 <| by simp [mul_assoc]
                                        /-
                                          🎉 no goals
                                        -/


@[simp]
theorem inv_I : I⁻¹ = -I := by
  /-
    ⊢ Eq (Inv.inv Complex.I) (Neg.neg Complex.I)
  -/
  rw [inv_eq_one_div, div_I, one_mul]
  /-
    🎉 no goals
  -/

-- @[simp]
/- Porting note (https://github.com/leanprover-community/mathlib4/issues/11119): `simp` attribute removed as linter reports this can be proved
by `simp only [@map_inv₀]` -/

theorem normSq_inv (z : ℂ) : normSq z⁻¹ = (normSq z)⁻¹ :=
  map_inv₀ normSq z

-- @[simp]
/- Porting note (https://github.com/leanprover-community/mathlib4/issues/11119): `simp` attribute removed as linter reports this can be proved
by `simp only [@map_div₀]` -/

theorem normSq_div (z w : ℂ) : normSq (z / w) = normSq z / normSq w :=
  map_div₀ normSq z w


lemma div_ofReal (z : ℂ) (x : ℝ) : z / x = ⟨z.re / x, z.im / x⟩ := by
  /-
    z : Complex
    x : Real
    ⊢ Eq (HDiv.hDiv z ↑x) { re := HDiv.hDiv z.re x, im := HDiv.hDiv z.im x }
  -/
  simp_rw [div_eq_inv_mul, ← ofReal_inv, ofReal_mul']
  /-
    🎉 no goals
  -/


lemma div_natCast (z : ℂ) (n : ℕ) : z / n = ⟨z.re / n, z.im / n⟩ :=
  mod_cast div_ofReal z n


@[deprecated (since := "2024-04-17")]
alias div_nat_cast := div_natCast


lemma div_intCast (z : ℂ) (n : ℤ) : z / n = ⟨z.re / n, z.im / n⟩ :=
  mod_cast div_ofReal z n


@[deprecated (since := "2024-04-17")]
alias div_int_cast := div_intCast


lemma div_ratCast (z : ℂ) (x : ℚ) : z / x = ⟨z.re / x, z.im / x⟩ :=
  mod_cast div_ofReal z x


@[deprecated (since := "2024-04-17")]
alias div_rat_cast := div_ratCast


lemma div_ofNat (z : ℂ) (n : ℕ) [n.AtLeastTwo] :
    z / ofNat(n) = ⟨z.re / ofNat(n), z.im / ofNat(n)⟩ :=
  div_natCast z n


                                                                          /-
                                                                            z : Complex
                                                                            x : Real
                                                                            ⊢ Eq (HDiv.hDiv z ↑x).re (HDiv.hDiv z.re x)
                                                                          -/
@[simp] lemma div_ofReal_re (z : ℂ) (x : ℝ) : (z / x).re = z.re / x := by rw [div_ofReal]
                                                                          /-
                                                                            🎉 no goals
                                                                          -/

                                                                          /-
                                                                            z : Complex
                                                                            x : Real
                                                                            ⊢ Eq (HDiv.hDiv z ↑x).im (HDiv.hDiv z.im x)
                                                                          -/
@[simp] lemma div_ofReal_im (z : ℂ) (x : ℝ) : (z / x).im = z.im / x := by rw [div_ofReal]
                                                                          /-
                                                                            🎉 no goals
                                                                          -/

                                                                           /-
                                                                             z : Complex
                                                                             n : Nat
                                                                             ⊢ Eq (HDiv.hDiv z ↑n).re (HDiv.hDiv z.re ↑n)
                                                                           -/
@[simp] lemma div_natCast_re (z : ℂ) (n : ℕ) : (z / n).re = z.re / n := by rw [div_natCast]
                                                                           /-
                                                                             🎉 no goals
                                                                           -/

                                                                           /-
                                                                             z : Complex
                                                                             n : Nat
                                                                             ⊢ Eq (HDiv.hDiv z ↑n).im (HDiv.hDiv z.im ↑n)
                                                                           -/
@[simp] lemma div_natCast_im (z : ℂ) (n : ℕ) : (z / n).im = z.im / n := by rw [div_natCast]
                                                                           /-
                                                                             🎉 no goals
                                                                           -/

                                                                           /-
                                                                             z : Complex
                                                                             n : Int
                                                                             ⊢ Eq (HDiv.hDiv z ↑n).re (HDiv.hDiv z.re ↑n)
                                                                           -/
@[simp] lemma div_intCast_re (z : ℂ) (n : ℤ) : (z / n).re = z.re / n := by rw [div_intCast]
                                                                           /-
                                                                             🎉 no goals
                                                                           -/

                                                                           /-
                                                                             z : Complex
                                                                             n : Int
                                                                             ⊢ Eq (HDiv.hDiv z ↑n).im (HDiv.hDiv z.im ↑n)
                                                                           -/
@[simp] lemma div_intCast_im (z : ℂ) (n : ℤ) : (z / n).im = z.im / n := by rw [div_intCast]
                                                                           /-
                                                                             🎉 no goals
                                                                           -/

                                                                           /-
                                                                             z : Complex
                                                                             x : Rat
                                                                             ⊢ Eq (HDiv.hDiv z ↑x).re (HDiv.hDiv z.re ↑x)
                                                                           -/
@[simp] lemma div_ratCast_re (z : ℂ) (x : ℚ) : (z / x).re = z.re / x := by rw [div_ratCast]
                                                                           /-
                                                                             🎉 no goals
                                                                           -/

                                                                           /-
                                                                             z : Complex
                                                                             x : Rat
                                                                             ⊢ Eq (HDiv.hDiv z ↑x).im (HDiv.hDiv z.im ↑x)
                                                                           -/
@[simp] lemma div_ratCast_im (z : ℂ) (x : ℚ) : (z / x).im = z.im / x := by rw [div_ratCast]
                                                                           /-
                                                                             🎉 no goals
                                                                           -/


@[deprecated (since := "2024-04-17")]
alias div_rat_cast_im := div_ratCast_im


@[simp]
lemma div_ofNat_re (z : ℂ) (n : ℕ) [n.AtLeastTwo] :
    (z / ofNat(n)).re = z.re / ofNat(n) := div_natCast_re z n


@[simp]
lemma div_ofNat_im (z : ℂ) (n : ℕ) [n.AtLeastTwo] :
    (z / ofNat(n)).im = z.im / ofNat(n) := div_natCast_im z n


instance instCharZero : CharZero ℂ :=
                                     /-
                                       α : Type u_1
                                       n : Nat
                                       h : Eq (↑n) 0
                                       ⊢ Eq n 0
                                     -/
  charZero_of_inj_zero fun n h => by rwa [← ofReal_natCast, ofReal_eq_zero, Nat.cast_eq_zero] at h
                                     /-
                                       🎉 no goals
                                     -/


/-- A complex number `z` plus its conjugate `conj z` is `2` times its real part. -/
theorem re_eq_add_conj (z : ℂ) : (z.re : ℂ) = (z + conj z) / 2 := by
  /-
    z : Complex
    ⊢ Eq (↑z.re) (HDiv.hDiv (HAdd.hAdd z ((starRingEnd Complex) z)) 2)
  -/
  simp only [add_conj, ofReal_mul, ofReal_ofNat, mul_div_cancel_left₀ (z.re : ℂ) two_ne_zero]
  /-
    🎉 no goals
  -/


/-- A complex number `z` minus its conjugate `conj z` is `2i` times its imaginary part. -/
theorem im_eq_sub_conj (z : ℂ) : (z.im : ℂ) = (z - conj z) / (2 * I) := by
  simp only [sub_conj, ofReal_mul, ofReal_ofNat, mul_right_comm,
    mul_div_cancel_left₀ _ (mul_ne_zero two_ne_zero I_ne_zero : 2 * I ≠ 0)]


/-- Show the imaginary number ⟨x, y⟩ as an "x + y*I" string

Note that the Real numbers used for x and y will show as cauchy sequences due to the way Real
numbers are represented.
-/
unsafe instance instRepr : Repr ℂ where
  reprPrec f p :=
    (if p > 65 then (Std.Format.bracket "(" · ")") else (·)) <|
      reprPrec f.re 65 ++ " + " ++ reprPrec f.im 70 ++ "*I"


/-- The preimage under `equivRealProd` of `s ×ˢ t` is `s ×ℂ t`. -/
lemma preimage_equivRealProd_prod (s t : Set ℝ) : equivRealProd ⁻¹' (s ×ˢ t) = s ×ℂ t := rfl


/-- The inequality `s × t ⊆ s₁ × t₁` holds in `ℂ` iff it holds in `ℝ × ℝ`. -/
lemma reProdIm_subset_iff {s s₁ t t₁ : Set ℝ} : s ×ℂ t ⊆ s₁ ×ℂ t₁ ↔ s ×ˢ t ⊆ s₁ ×ˢ t₁ := by
  /-
    s s₁ t t₁ : Set Real
    ⊢ Iff (HasSubset.Subset (Complex.reProdIm s t) (Complex.reProdIm s₁ t₁)) (HasS …
  -/
  rw [← @preimage_equivRealProd_prod s t, ← @preimage_equivRealProd_prod s₁ t₁]
  /-
    s s₁ t t₁ : Set Real
    ⊢ Iff (HasSubset.Subset (Set.preimage (⇑Complex.equivRealProd) (SProd.sprod s  …
  -/
  exact Equiv.preimage_subset equivRealProd _ _
  /-
    🎉 no goals
  -/


/-- If `s ⊆ s₁ ⊆ ℝ` and `t ⊆ t₁ ⊆ ℝ`, then `s × t ⊆ s₁ × t₁` in `ℂ`. -/
lemma reProdIm_subset_iff' {s s₁ t t₁ : Set ℝ} :
    s ×ℂ t ⊆ s₁ ×ℂ t₁ ↔ s ⊆ s₁ ∧ t ⊆ t₁ ∨ s = ∅ ∨ t = ∅ := by
  /-
    s s₁ t t₁ : Set Real
    ⊢ Iff (HasSubset.Subset (Complex.reProdIm s t) (Complex.reProdIm s₁ t₁)) (Or ( …
  -/
  convert prod_subset_prod_iff
  /-
    case h.e'_1.a
    s s₁ t t₁ : Set Real
    ⊢ Iff (HasSubset.Subset (Complex.reProdIm s t) (Complex.reProdIm s₁ t₁)) (HasS …
  -/
  exact reProdIm_subset_iff
  /-
    🎉 no goals
  -/


@[simp] lemma reProdIm_nonempty : (s ×ℂ t).Nonempty ↔ s.Nonempty ∧ t.Nonempty := by
  /-
    s t : Set Real
    ⊢ Iff (Complex.reProdIm s t).Nonempty (And s.Nonempty t.Nonempty)
  -/
  simp [Set.Nonempty, reProdIm, Complex.exists]
  /-
    🎉 no goals
  -/


@[simp] lemma reProdIm_eq_empty : s ×ℂ t = ∅ ↔ s = ∅ ∨ t = ∅ := by
  /-
    s t : Set Real
    ⊢ Iff (Eq (Complex.reProdIm s t) EmptyCollection.emptyCollection) (Or (Eq s Em …
  -/
  simp [← not_nonempty_iff_eq_empty, reProdIm_nonempty, -not_and, not_and_or]
  /-
    🎉 no goals
  -/


/-- A `Rectangle` is an axis-parallel rectangle with corners `z` and `w`. -/
def Rectangle (z w : ℂ) : Set ℂ := [[z.re, w.re]] ×ℂ [[z.im, w.im]]


/-- A real segment `[a₁, a₂]` translated by `b * I` is the complex line segment. -/
lemma horizontalSegment_eq (a₁ a₂ b : ℝ) :
    (fun (x : ℝ) ↦ x + b * I) '' [[a₁, a₂]] = [[a₁, a₂]] ×ℂ {b} := by
  /-
    a₁ a₂ b : Real
    ⊢ Eq (Set.image (fun x => HAdd.hAdd (↑x) (HMul.hMul (↑b) Complex.I)) (Set.uIcc …
  -/
  rw [← preimage_equivRealProd_prod]
  /-
    a₁ a₂ b : Real
    ⊢ Eq (Set.image (fun x => HAdd.hAdd (↑x) (HMul.hMul (↑b) Complex.I)) (Set.uIcc …
  -/
  ext x
  /-
    case h
    a₁ a₂ b : Real
    x : Complex
    ⊢ Iff (Membership.mem (Set.image (fun x => HAdd.hAdd (↑x) (HMul.hMul (↑b) Comp …
  -/
  constructor
    /-
      case h.mp
      a₁ a₂ b : Real
      x : Complex
      ⊢ Membership.mem (Set.image (fun x => HAdd.hAdd (↑x) (HMul.hMul (↑b) Complex.I …
    -/
  · intro hx
    /-
      case h.mp
      a₁ a₂ b : Real
      x : Complex
      hx : Membership.mem (Set.image (fun x => HAdd.hAdd (↑x) (HMul.hMul (↑b) Comple …
      ⊢ Membership.mem (Set.preimage (⇑Complex.equivRealProd) (SProd.sprod (Set.uIcc …
    -/
    obtain ⟨x₁, hx₁, hx₁'⟩ := hx
    /-
      case h.mp.intro.intro
      a₁ a₂ b : Real
      x : Complex
      x₁ : Real
      hx₁ : Membership.mem (Set.uIcc a₁ a₂) x₁
      hx₁' : Eq ((fun x => HAdd.hAdd (↑x) (HMul.hMul (↑b) Complex.I)) x₁) x
      ⊢ Membership.mem (Set.preimage (⇑Complex.equivRealProd) (SProd.sprod (Set.uIcc …
    -/
    simp [← hx₁', mem_preimage, mem_prod, hx₁]
    /-
      🎉 no goals
    -/
    /-
      case h.mpr
      a₁ a₂ b : Real
      x : Complex
      ⊢ Membership.mem (Set.preimage (⇑Complex.equivRealProd) (SProd.sprod (Set.uIcc …
    -/
  · intro hx
    /-
      case h.mpr
      a₁ a₂ b : Real
      x : Complex
      hx : Membership.mem (Set.preimage (⇑Complex.equivRealProd) (SProd.sprod (Set.u …
      ⊢ Membership.mem (Set.image (fun x => HAdd.hAdd (↑x) (HMul.hMul (↑b) Complex.I …
    -/
    obtain ⟨x₁, hx₁, hx₁', hx₁''⟩ := hx
    /-
      case h.mpr.intro.refl
      a₁ a₂ : Real
      x : Complex
      x₁ : Membership.mem (Set.uIcc a₁ a₂) (Complex.equivRealProd x).1
      ⊢ Membership.mem (Set.image (fun x_1 => HAdd.hAdd (↑x_1) (HMul.hMul (↑(Complex …
    -/
    refine ⟨x.re, x₁, by simp⟩
    /-
      🎉 no goals
    -/


/-- A vertical segment `[b₁, b₂]` translated by `a` is the complex line segment. -/
lemma verticalSegment_eq (a b₁ b₂ : ℝ) :
    (fun (y : ℝ) ↦ a + y * I) '' [[b₁, b₂]] = {a} ×ℂ [[b₁, b₂]] := by
  /-
    a b₁ b₂ : Real
    ⊢ Eq (Set.image (fun y => HAdd.hAdd (↑a) (HMul.hMul (↑y) Complex.I)) (Set.uIcc …
  -/
  rw [← preimage_equivRealProd_prod]
  /-
    a b₁ b₂ : Real
    ⊢ Eq (Set.image (fun y => HAdd.hAdd (↑a) (HMul.hMul (↑y) Complex.I)) (Set.uIcc …
  -/
  ext x
  /-
    case h
    a b₁ b₂ : Real
    x : Complex
    ⊢ Iff (Membership.mem (Set.image (fun y => HAdd.hAdd (↑a) (HMul.hMul (↑y) Comp …
  -/
  constructor
    /-
      case h.mp
      a b₁ b₂ : Real
      x : Complex
      ⊢ Membership.mem (Set.image (fun y => HAdd.hAdd (↑a) (HMul.hMul (↑y) Complex.I …
    -/
  · intro hx
    /-
      case h.mp
      a b₁ b₂ : Real
      x : Complex
      hx : Membership.mem (Set.image (fun y => HAdd.hAdd (↑a) (HMul.hMul (↑y) Comple …
      ⊢ Membership.mem (Set.preimage (⇑Complex.equivRealProd) (SProd.sprod (Singleto …
    -/
    obtain ⟨x₁, hx₁, hx₁'⟩ := hx
    /-
      case h.mp.intro.intro
      a b₁ b₂ : Real
      x : Complex
      x₁ : Real
      hx₁ : Membership.mem (Set.uIcc b₁ b₂) x₁
      hx₁' : Eq ((fun y => HAdd.hAdd (↑a) (HMul.hMul (↑y) Complex.I)) x₁) x
      ⊢ Membership.mem (Set.preimage (⇑Complex.equivRealProd) (SProd.sprod (Singleto …
    -/
    simp [← hx₁', mem_preimage, mem_prod, hx₁]
    /-
      🎉 no goals
    -/
    /-
      case h.mpr
      a b₁ b₂ : Real
      x : Complex
      ⊢ Membership.mem (Set.preimage (⇑Complex.equivRealProd) (SProd.sprod (Singleto …
    -/
  · intro hx
    simp only [equivRealProd_apply, singleton_prod, mem_image, Prod.mk.injEq,
      exists_eq_right_right, mem_preimage] at hx
    /-
      case h.mpr
      a b₁ b₂ : Real
      x : Complex
      hx : And (Membership.mem (Set.uIcc b₁ b₂) x.im) (Eq a x.re)
      ⊢ Membership.mem (Set.image (fun y => HAdd.hAdd (↑a) (HMul.hMul (↑y) Complex.I …
    -/
    obtain ⟨x₁, hx₁, hx₁', hx₁''⟩ := hx
    /-
      case h.mpr.intro.refl
      b₁ b₂ : Real
      x : Complex
      x₁ : Membership.mem (Set.uIcc b₁ b₂) x.im
      ⊢ Membership.mem (Set.image (fun y => HAdd.hAdd (↑x.re) (HMul.hMul (↑y) Comple …
    -/
    refine ⟨x.im, x₁, by simp⟩
    /-
      🎉 no goals
    -/


