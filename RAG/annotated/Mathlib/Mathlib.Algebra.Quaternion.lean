/-- Quaternion algebra over a type with fixed coefficients $a=i^2$ and $b=j^2$.
Implemented as a structure with four fields: `re`, `imI`, `imJ`, and `imK`. -/
@[ext]
structure QuaternionAlgebra (R : Type*) (a b : R) where
  /-- Real part of a quaternion. -/
  re : R
  /-- First imaginary part (i) of a quaternion. -/
  imI : R
  /-- Second imaginary part (j) of a quaternion. -/
  imJ : R
  /-- Third imaginary part (k) of a quaternion. -/
  imK : R


@[inherit_doc]
scoped[Quaternion] notation "ℍ[" R "," a "," b "]" => QuaternionAlgebra R a b

/-- The equivalence between a quaternion algebra over `R` and `R × R × R × R`. -/
@[simps]
def equivProd {R : Type*} (c₁ c₂ : R) : ℍ[R,c₁,c₂] ≃ R × R × R × R where
  toFun a := ⟨a.1, a.2, a.3, a.4⟩
  invFun a := ⟨a.1, a.2.1, a.2.2.1, a.2.2.2⟩
  left_inv _ := rfl
  right_inv _ := rfl


/-- The equivalence between a quaternion algebra over `R` and `Fin 4 → R`. -/
@[simps symm_apply]
def equivTuple {R : Type*} (c₁ c₂ : R) : ℍ[R,c₁,c₂] ≃ (Fin 4 → R) where
  toFun a := ![a.1, a.2, a.3, a.4]
  invFun a := ⟨a 0, a 1, a 2, a 3⟩
  left_inv _ := rfl
                    /-
                      R : Type u_1
                      c₁ c₂ : R
                      f : Fin 4 → R
                      ⊢ Eq ((fun a => Matrix.vecCons a.re (Matrix.vecCons a.imI (Matrix.vecCons a.im …
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
  right_inv f := by ext ⟨_, _ | _ | _ | _ | _ | ⟨⟩⟩ <;> rfl
                                                        /-
                                                          🎉 no goals
                                                        -/


@[simp]
theorem equivTuple_apply {R : Type*} (c₁ c₂ : R) (x : ℍ[R,c₁,c₂]) :
    equivTuple c₁ c₂ x = ![x.re, x.imI, x.imJ, x.imK] :=
  rfl


@[simp]
theorem mk.eta {R : Type*} {c₁ c₂} (a : ℍ[R,c₁,c₂]) : mk a.1 a.2 a.3 a.4 = a := rfl


instance [Subsingleton R] : Subsingleton ℍ[R, c₁, c₂] := (equivTuple c₁ c₂).subsingleton

instance [Nontrivial R] : Nontrivial ℍ[R, c₁, c₂] := (equivTuple c₁ c₂).surjective.nontrivial


/-- The imaginary part of a quaternion. -/
def im (x : ℍ[R,c₁,c₂]) : ℍ[R,c₁,c₂] :=
  ⟨0, x.imI, x.imJ, x.imK⟩


@[simp]
theorem im_re : a.im.re = 0 :=
  rfl


@[simp]
theorem im_imI : a.im.imI = a.imI :=
  rfl


@[simp]
theorem im_imJ : a.im.imJ = a.imJ :=
  rfl


@[simp]
theorem im_imK : a.im.imK = a.imK :=
  rfl


@[simp]
theorem im_idem : a.im.im = a.im :=
  rfl


/-- Coercion `R → ℍ[R,c₁,c₂]`. -/
@[coe] def coe (x : R) : ℍ[R,c₁,c₂] := ⟨x, 0, 0, 0⟩


instance : CoeTC R ℍ[R,c₁,c₂] := ⟨coe⟩


@[simp, norm_cast]
theorem coe_re : (x : ℍ[R,c₁,c₂]).re = x := rfl


@[simp, norm_cast]
theorem coe_imI : (x : ℍ[R,c₁,c₂]).imI = 0 := rfl


@[simp, norm_cast]
theorem coe_imJ : (x : ℍ[R,c₁,c₂]).imJ = 0 := rfl


@[simp, norm_cast]
theorem coe_imK : (x : ℍ[R,c₁,c₂]).imK = 0 := rfl


theorem coe_injective : Function.Injective (coe : R → ℍ[R,c₁,c₂]) := fun _ _ h => congr_arg re h


@[simp]
theorem coe_inj {x y : R} : (x : ℍ[R,c₁,c₂]) = y ↔ x = y :=
  coe_injective.eq_iff

-- Porting note: removed `simps`, added simp lemmas manually. Should adjust `simps` to name properly

instance : Zero ℍ[R,c₁,c₂] := ⟨⟨0, 0, 0, 0⟩⟩


@[simp] theorem zero_re : (0 : ℍ[R,c₁,c₂]).re = 0 := rfl


@[simp] theorem zero_imI : (0 : ℍ[R,c₁,c₂]).imI = 0 := rfl


@[simp] theorem zero_imJ : (0 : ℍ[R,c₁,c₂]).imJ = 0 := rfl


@[simp] theorem zero_imK : (0 : ℍ[R,c₁,c₂]).imK = 0 := rfl


@[simp] theorem zero_im : (0 : ℍ[R,c₁,c₂]).im = 0 := rfl


@[simp, norm_cast]
theorem coe_zero : ((0 : R) : ℍ[R,c₁,c₂]) = 0 := rfl


instance : Inhabited ℍ[R,c₁,c₂] := ⟨0⟩


instance : One ℍ[R,c₁,c₂] := ⟨⟨1, 0, 0, 0⟩⟩


@[simp] theorem one_re : (1 : ℍ[R,c₁,c₂]).re = 1 := rfl


@[simp] theorem one_imI : (1 : ℍ[R,c₁,c₂]).imI = 0 := rfl


@[simp] theorem one_imJ : (1 : ℍ[R,c₁,c₂]).imJ = 0 := rfl


@[simp] theorem one_imK : (1 : ℍ[R,c₁,c₂]).imK = 0 := rfl


@[simp] theorem one_im : (1 : ℍ[R,c₁,c₂]).im = 0 := rfl


@[simp, norm_cast]
theorem coe_one : ((1 : R) : ℍ[R,c₁,c₂]) = 1 := rfl


instance : Add ℍ[R,c₁,c₂] :=
  ⟨fun a b => ⟨a.1 + b.1, a.2 + b.2, a.3 + b.3, a.4 + b.4⟩⟩


@[simp] theorem add_re : (a + b).re = a.re + b.re := rfl


@[simp] theorem add_imI : (a + b).imI = a.imI + b.imI := rfl


@[simp] theorem add_imJ : (a + b).imJ = a.imJ + b.imJ := rfl


@[simp] theorem add_imK : (a + b).imK = a.imK + b.imK := rfl


@[simp]
theorem mk_add_mk (a₁ a₂ a₃ a₄ b₁ b₂ b₃ b₄ : R) :
    (mk a₁ a₂ a₃ a₄ : ℍ[R,c₁,c₂]) + mk b₁ b₂ b₃ b₄ = mk (a₁ + b₁) (a₂ + b₂) (a₃ + b₃) (a₄ + b₄) :=
  rfl


@[simp] theorem add_im : (a + b).im = a.im + b.im :=
  QuaternionAlgebra.ext (zero_add _).symm rfl rfl rfl


@[simp, norm_cast]
                                                           /-
                                                             R : Type u_3
                                                             c₁ c₂ x y : R
                                                             inst✝ : AddZeroClass R
                                                             ⊢ Eq (↑(HAdd.hAdd x y)) (HAdd.hAdd ↑x ↑y)
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
theorem coe_add : ((x + y : R) : ℍ[R,c₁,c₂]) = x + y := by ext <;> simp
                                                                   /-
                                                                     🎉 no goals
                                                                   -/


instance : Neg ℍ[R,c₁,c₂] := ⟨fun a => ⟨-a.1, -a.2, -a.3, -a.4⟩⟩


@[simp] theorem neg_re : (-a).re = -a.re := rfl


@[simp] theorem neg_imI : (-a).imI = -a.imI := rfl


@[simp] theorem neg_imJ : (-a).imJ = -a.imJ := rfl


@[simp] theorem neg_imK : (-a).imK = -a.imK := rfl


@[simp]
theorem neg_mk (a₁ a₂ a₃ a₄ : R) : -(mk a₁ a₂ a₃ a₄ : ℍ[R,c₁,c₂]) = ⟨-a₁, -a₂, -a₃, -a₄⟩ :=
  rfl


@[simp] theorem neg_im : (-a).im = -a.im :=
  QuaternionAlgebra.ext neg_zero.symm rfl rfl rfl


@[simp, norm_cast]
                                                     /-
                                                       R : Type u_3
                                                       c₁ c₂ x : R
                                                       inst✝ : AddGroup R
                                                       ⊢ Eq (↑(Neg.neg x)) (Neg.neg ↑x)
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
theorem coe_neg : ((-x : R) : ℍ[R,c₁,c₂]) = -x := by ext <;> simp
                                                             /-
                                                               🎉 no goals
                                                             -/


instance : Sub ℍ[R,c₁,c₂] :=
  ⟨fun a b => ⟨a.1 - b.1, a.2 - b.2, a.3 - b.3, a.4 - b.4⟩⟩


@[simp] theorem sub_re : (a - b).re = a.re - b.re := rfl


@[simp] theorem sub_imI : (a - b).imI = a.imI - b.imI := rfl


@[simp] theorem sub_imJ : (a - b).imJ = a.imJ - b.imJ := rfl


@[simp] theorem sub_imK : (a - b).imK = a.imK - b.imK := rfl


@[simp] theorem sub_im : (a - b).im = a.im - b.im :=
  QuaternionAlgebra.ext (sub_zero _).symm rfl rfl rfl


@[simp]
theorem mk_sub_mk (a₁ a₂ a₃ a₄ b₁ b₂ b₃ b₄ : R) :
    (mk a₁ a₂ a₃ a₄ : ℍ[R,c₁,c₂]) - mk b₁ b₂ b₃ b₄ = mk (a₁ - b₁) (a₂ - b₂) (a₃ - b₃) (a₄ - b₄) :=
  rfl


@[simp, norm_cast]
theorem coe_im : (x : ℍ[R,c₁,c₂]).im = 0 :=
  rfl


@[simp]
theorem re_add_im : ↑a.re + a.im = a :=
  QuaternionAlgebra.ext (add_zero _) (zero_add _) (zero_add _) (zero_add _)


@[simp]
theorem sub_self_im : a - a.im = a.re :=
  QuaternionAlgebra.ext (sub_zero _) (sub_self _) (sub_self _) (sub_self _)


@[simp]
theorem sub_self_re : a - a.re = a.im :=
  QuaternionAlgebra.ext (sub_self _) (sub_zero _) (sub_zero _) (sub_zero _)


/-- Multiplication is given by

* `1 * x = x * 1 = x`;
* `i * i = c₁`;
* `j * j = c₂`;
* `i * j = k`, `j * i = -k`;
* `k * k = -c₁ * c₂`;
* `i * k = c₁ * j`, `k * i = -c₁ * j`;
* `j * k = -c₂ * i`, `k * j = c₂ * i`. -/
instance : Mul ℍ[R,c₁,c₂] :=
  ⟨fun a b =>
    ⟨a.1 * b.1 + c₁ * a.2 * b.2 + c₂ * a.3 * b.3 - c₁ * c₂ * a.4 * b.4,
      a.1 * b.2 + a.2 * b.1 - c₂ * a.3 * b.4 + c₂ * a.4 * b.3,
      a.1 * b.3 + c₁ * a.2 * b.4 + a.3 * b.1 - c₁ * a.4 * b.2,
      a.1 * b.4 + a.2 * b.3 - a.3 * b.2 + a.4 * b.1⟩⟩


@[simp]
theorem mul_re : (a * b).re = a.1 * b.1 + c₁ * a.2 * b.2 + c₂ * a.3 * b.3 - c₁ * c₂ * a.4 * b.4 :=
  rfl


@[simp]
theorem mul_imI : (a * b).imI = a.1 * b.2 + a.2 * b.1 - c₂ * a.3 * b.4 + c₂ * a.4 * b.3 := rfl


@[simp]
theorem mul_imJ : (a * b).imJ = a.1 * b.3 + c₁ * a.2 * b.4 + a.3 * b.1 - c₁ * a.4 * b.2 := rfl


@[simp] theorem mul_imK : (a * b).imK = a.1 * b.4 + a.2 * b.3 - a.3 * b.2 + a.4 * b.1 := rfl


@[simp]
theorem mk_mul_mk (a₁ a₂ a₃ a₄ b₁ b₂ b₃ b₄ : R) :
    (mk a₁ a₂ a₃ a₄ : ℍ[R,c₁,c₂]) * mk b₁ b₂ b₃ b₄ =
      ⟨a₁ * b₁ + c₁ * a₂ * b₂ + c₂ * a₃ * b₃ - c₁ * c₂ * a₄ * b₄,
        a₁ * b₂ + a₂ * b₁ - c₂ * a₃ * b₄ + c₂ * a₄ * b₃,
        a₁ * b₃ + c₁ * a₂ * b₄ + a₃ * b₁ - c₁ * a₄ * b₂, a₁ * b₄ + a₂ * b₃ - a₃ * b₂ + a₄ * b₁⟩ :=
  rfl


instance : SMul S ℍ[R,c₁,c₂] where smul s a := ⟨s • a.1, s • a.2, s • a.3, s • a.4⟩


instance [SMul S T] [IsScalarTower S T R] : IsScalarTower S T ℍ[R,c₁,c₂] where
                         /-
                           S : Type u_1
                           T : Type u_2
                           R : Type u_3
                           c₁ c₂ r x✝ y : R
                           a b : QuaternionAlgebra R c₁ c₂
                           inst✝³ : SMul S R
                           inst✝² : SMul T R
                           s✝ : S
                           inst✝¹ : SMul S T
                           inst✝ : IsScalarTower S T R
                           s : S
                           t : T
                           x : QuaternionAlgebra R c₁ c₂
                           ⊢ Eq (HSMul.hSMul (HSMul.hSMul s t) x) (HSMul.hSMul s (HSMul.hSMul t x))
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
  smul_assoc s t x := by ext <;> exact smul_assoc _ _ _
                                 /-
                                   🎉 no goals
                                 -/


instance [SMulCommClass S T R] : SMulCommClass S T ℍ[R,c₁,c₂] where
                        /-
                          S : Type u_1
                          T : Type u_2
                          R : Type u_3
                          c₁ c₂ r x✝ y : R
                          a b : QuaternionAlgebra R c₁ c₂
                          inst✝² : SMul S R
                          inst✝¹ : SMul T R
                          s✝ : S
                          inst✝ : SMulCommClass S T R
                          s : S
                          t : T
                          x : QuaternionAlgebra R c₁ c₂
                          ⊢ Eq (HSMul.hSMul s (HSMul.hSMul t x)) (HSMul.hSMul t (HSMul.hSMul s x))
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
  smul_comm s t x := by ext <;> exact smul_comm _ _ _
                                /-
                                  🎉 no goals
                                -/


@[simp] theorem smul_re : (s • a).re = s • a.re := rfl


@[simp] theorem smul_imI : (s • a).imI = s • a.imI := rfl


@[simp] theorem smul_imJ : (s • a).imJ = s • a.imJ := rfl


@[simp] theorem smul_imK : (s • a).imK = s • a.imK := rfl


@[simp] theorem smul_im {S} [CommRing R] [SMulZeroClass S R] (s : S) : (s • a).im = s • a.im :=
  QuaternionAlgebra.ext (smul_zero s).symm rfl rfl rfl


@[simp]
theorem smul_mk (re im_i im_j im_k : R) :
    s • (⟨re, im_i, im_j, im_k⟩ : ℍ[R,c₁,c₂]) = ⟨s • re, s • im_i, s • im_j, s • im_k⟩ :=
  rfl


@[simp, norm_cast]
theorem coe_smul [Zero R] [SMulZeroClass S R] (s : S) (r : R) :
    (↑(s • r) : ℍ[R,c₁,c₂]) = s • (r : ℍ[R,c₁,c₂]) :=
  QuaternionAlgebra.ext rfl (smul_zero s).symm (smul_zero s).symm (smul_zero s).symm


instance [AddCommGroup R] : AddCommGroup ℍ[R,c₁,c₂] :=
  (equivProd c₁ c₂).injective.addCommGroup _ rfl (fun _ _ ↦ rfl) (fun _ ↦ rfl) (fun _ _ ↦ rfl)
    (fun _ _ ↦ rfl) (fun _ _ ↦ rfl)


instance : AddCommGroupWithOne ℍ[R,c₁,c₂] where
  natCast n := ((n : R) : ℍ[R,c₁,c₂])
                     /-
                       S : Type u_1
                       T : Type u_2
                       R : Type u_3
                       c₁ c₂ r x y : R
                       a b : QuaternionAlgebra R c₁ c₂
                       inst✝ : AddCommGroupWithOne R
                       ⊢ Eq (NatCast.natCast 0) 0
                     -/
  natCast_zero := by simp
                     /-
                       🎉 no goals
                     -/
                     /-
                       S : Type u_1
                       T : Type u_2
                       R : Type u_3
                       c₁ c₂ r x y : R
                       a b : QuaternionAlgebra R c₁ c₂
                       inst✝ : AddCommGroupWithOne R
                       ⊢ ∀ (n : Nat), Eq (NatCast.natCast (HAdd.hAdd n 1)) (HAdd.hAdd (NatCast.natCas …
                     -/
  natCast_succ := by simp
                     /-
                       🎉 no goals
                     -/
  intCast n := ((n : R) : ℍ[R,c₁,c₂])
  intCast_ofNat _ := congr_arg coe (Int.cast_natCast _)
  intCast_negSucc n := by
    /-
      S : Type u_1
      T : Type u_2
      R : Type u_3
      c₁ c₂ r x y : R
      a b : QuaternionAlgebra R c₁ c₂
      inst✝ : AddCommGroupWithOne R
      n : Nat
      ⊢ Eq (IntCast.intCast (Int.negSucc n)) (Neg.neg ↑(HAdd.hAdd n 1))
    -/
    change coe _ = -coe _
    /-
      S : Type u_1
      T : Type u_2
      R : Type u_3
      c₁ c₂ r x y : R
      a b : QuaternionAlgebra R c₁ c₂
      inst✝ : AddCommGroupWithOne R
      n : Nat
      ⊢ Eq (↑↑(Int.negSucc n)) (Neg.neg ↑↑(HAdd.hAdd n 1))
    -/
    rw [Int.cast_negSucc, coe_neg]
    /-
      🎉 no goals
    -/


@[simp, norm_cast]
theorem natCast_re (n : ℕ) : (n : ℍ[R,c₁,c₂]).re = n :=
  rfl


@[deprecated (since := "2024-04-17")]
alias nat_cast_re := natCast_re


@[simp, norm_cast]
theorem natCast_imI (n : ℕ) : (n : ℍ[R,c₁,c₂]).imI = 0 :=
  rfl


@[deprecated (since := "2024-04-17")]
alias nat_cast_imI := natCast_imI


@[simp, norm_cast]
theorem natCast_imJ (n : ℕ) : (n : ℍ[R,c₁,c₂]).imJ = 0 :=
  rfl


@[deprecated (since := "2024-04-17")]
alias nat_cast_imJ := natCast_imJ


@[simp, norm_cast]
theorem natCast_imK (n : ℕ) : (n : ℍ[R,c₁,c₂]).imK = 0 :=
  rfl


@[deprecated (since := "2024-04-17")]
alias nat_cast_imK := natCast_imK


@[simp, norm_cast]
theorem natCast_im (n : ℕ) : (n : ℍ[R,c₁,c₂]).im = 0 :=
  rfl


@[deprecated (since := "2024-04-17")]
alias nat_cast_im := natCast_im


@[norm_cast]
theorem coe_natCast (n : ℕ) : ↑(n : R) = (n : ℍ[R,c₁,c₂]) :=
  rfl


@[deprecated (since := "2024-04-17")]
alias coe_nat_cast := coe_natCast


@[simp, norm_cast]
theorem intCast_re (z : ℤ) : (z : ℍ[R,c₁,c₂]).re = z :=
  rfl


@[deprecated (since := "2024-04-17")]
alias int_cast_re := intCast_re


@[simp, norm_cast]
theorem intCast_imI (z : ℤ) : (z : ℍ[R,c₁,c₂]).imI = 0 :=
  rfl


@[deprecated (since := "2024-04-17")]
alias int_cast_imI := intCast_imI


@[simp, norm_cast]
theorem intCast_imJ (z : ℤ) : (z : ℍ[R,c₁,c₂]).imJ = 0 :=
  rfl


@[deprecated (since := "2024-04-17")]
alias int_cast_imJ := intCast_imJ


@[simp, norm_cast]
theorem intCast_imK (z : ℤ) : (z : ℍ[R,c₁,c₂]).imK = 0 :=
  rfl


@[deprecated (since := "2024-04-17")]
alias int_cast_imK := intCast_imK


@[simp, norm_cast]
theorem intCast_im (z : ℤ) : (z : ℍ[R,c₁,c₂]).im = 0 :=
  rfl


@[deprecated (since := "2024-04-17")]
alias int_cast_im := intCast_im


@[norm_cast]
theorem coe_intCast (z : ℤ) : ↑(z : R) = (z : ℍ[R,c₁,c₂]) :=
  rfl


@[deprecated (since := "2024-04-17")]
alias coe_int_cast := coe_intCast


instance instRing : Ring ℍ[R,c₁,c₂] where
  __ := inferInstanceAs (AddCommGroupWithOne ℍ[R,c₁,c₂])
                           /-
                             S : Type u_1
                             T : Type u_2
                             R : Type u_3
                             c₁ c₂ r x y : R
                             a b : QuaternionAlgebra R c₁ c₂
                             inst✝ : CommRing R
                             x✝² x✝¹ x✝ : QuaternionAlgebra R c₁ c₂
                             ⊢ Eq (HMul.hMul x✝² (HAdd.hAdd x✝¹ x✝)) (HAdd.hAdd (HMul.hMul x✝² x✝¹) (HMul.h …
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
  left_distrib _ _ _ := by ext <;> simp <;> ring
                                            /-
                                              🎉 no goals
                                            -/
                            /-
                              S : Type u_1
                              T : Type u_2
                              R : Type u_3
                              c₁ c₂ r x y : R
                              a b : QuaternionAlgebra R c₁ c₂
                              inst✝ : CommRing R
                              x✝² x✝¹ x✝ : QuaternionAlgebra R c₁ c₂
                              ⊢ Eq (HMul.hMul (HAdd.hAdd x✝² x✝¹) x✝) (HAdd.hAdd (HMul.hMul x✝² x✝) (HMul.hM …
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
  right_distrib _ _ _ := by ext <;> simp <;> ring
                                             /-
                                               🎉 no goals
                                             -/
                   /-
                     S : Type u_1
                     T : Type u_2
                     R : Type u_3
                     c₁ c₂ r x y : R
                     a b : QuaternionAlgebra R c₁ c₂
                     inst✝ : CommRing R
                     x✝ : QuaternionAlgebra R c₁ c₂
                     ⊢ Eq (HMul.hMul 0 x✝) 0
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
  zero_mul _ := by ext <;> simp
                           /-
                             🎉 no goals
                           -/
                   /-
                     S : Type u_1
                     T : Type u_2
                     R : Type u_3
                     c₁ c₂ r x y : R
                     a b : QuaternionAlgebra R c₁ c₂
                     inst✝ : CommRing R
                     x✝ : QuaternionAlgebra R c₁ c₂
                     ⊢ Eq (HMul.hMul x✝ 0) 0
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
  mul_zero _ := by ext <;> simp
                           /-
                             🎉 no goals
                           -/
                        /-
                          S : Type u_1
                          T : Type u_2
                          R : Type u_3
                          c₁ c₂ r x y : R
                          a b : QuaternionAlgebra R c₁ c₂
                          inst✝ : CommRing R
                          x✝² x✝¹ x✝ : QuaternionAlgebra R c₁ c₂
                          ⊢ Eq (HMul.hMul (HMul.hMul x✝² x✝¹) x✝) (HMul.hMul x✝² (HMul.hMul x✝¹ x✝))
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
  mul_assoc _ _ _ := by ext <;> simp <;> ring
                                         /-
                                           🎉 no goals
                                         -/
                  /-
                    S : Type u_1
                    T : Type u_2
                    R : Type u_3
                    c₁ c₂ r x y : R
                    a b : QuaternionAlgebra R c₁ c₂
                    inst✝ : CommRing R
                    x✝ : QuaternionAlgebra R c₁ c₂
                    ⊢ Eq (HMul.hMul 1 x✝) x✝
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
  one_mul _ := by ext <;> simp
                          /-
                            🎉 no goals
                          -/
                  /-
                    S : Type u_1
                    T : Type u_2
                    R : Type u_3
                    c₁ c₂ r x y : R
                    a b : QuaternionAlgebra R c₁ c₂
                    inst✝ : CommRing R
                    x✝ : QuaternionAlgebra R c₁ c₂
                    ⊢ Eq (HMul.hMul x✝ 1) x✝
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
  mul_one _ := by ext <;> simp
                          /-
                            🎉 no goals
                          -/


@[norm_cast, simp]
                                                           /-
                                                             R : Type u_3
                                                             c₁ c₂ x y : R
                                                             inst✝ : CommRing R
                                                             ⊢ Eq (↑(HMul.hMul x y)) (HMul.hMul ↑x ↑y)
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
theorem coe_mul : ((x * y : R) : ℍ[R,c₁,c₂]) = x * y := by ext <;> simp
                                                                   /-
                                                                     🎉 no goals
                                                                   -/

-- TODO: add weaker `MulAction`, `DistribMulAction`, and `Module` instances (and repeat them
-- for `ℍ[R]`)

instance [CommSemiring S] [Algebra S R] : Algebra S ℍ[R,c₁,c₂] where
  smul := (· • ·)
  toFun s := coe (algebraMap S R s)
                 /-
                   S : Type u_1
                   T : Type u_2
                   R : Type u_3
                   c₁ c₂ r x y : R
                   a b : QuaternionAlgebra R c₁ c₂
                   inst✝² : CommRing R
                   inst✝¹ : CommSemiring S
                   inst✝ : Algebra S R
                   ⊢ Eq ((fun s => ↑((algebraMap S R) s)) 1) 1
                 -/
  map_one' := by simp only [map_one, coe_one]
                 /-
                   🎉 no goals
                 -/
                  /-
                    S : Type u_1
                    T : Type u_2
                    R : Type u_3
                    c₁ c₂ r x y : R
                    a b : QuaternionAlgebra R c₁ c₂
                    inst✝² : CommRing R
                    inst✝¹ : CommSemiring S
                    inst✝ : Algebra S R
                    ⊢ Eq ((↑{ toFun := fun s => ↑((algebraMap S R) s), map_one' := ⋯, map_mul' :=  …
                  -/
                     /-
                       S : Type u_1
                       T : Type u_2
                       R : Type u_3
                       c₁ c₂ r x✝ y✝ : R
                       a b : QuaternionAlgebra R c₁ c₂
                       inst✝² : CommRing R
                       inst✝¹ : CommSemiring S
                       inst✝ : Algebra S R
                       x y : S
                       ⊢ Eq ({ toFun := fun s => ↑((algebraMap S R) s), map_one' := ⋯ }.toFun (HMul.h …
                     -/
  map_zero' := by simp only [map_zero, coe_zero]
                     /-
                       🎉 no goals
                     -/
                  /-
                    🎉 no goals
                  -/
  map_mul' x y := by simp only [map_mul, coe_mul]
                     /-
                       S : Type u_1
                       T : Type u_2
                       R : Type u_3
                       c₁ c₂ r x✝ y✝ : R
                       a b : QuaternionAlgebra R c₁ c₂
                       inst✝² : CommRing R
                       inst✝¹ : CommSemiring S
                       inst✝ : Algebra S R
                       x y : S
                       ⊢ Eq ((↑{ toFun := fun s => ↑((algebraMap S R) s), map_one' := ⋯, map_mul' :=  …
                     -/
  map_add' x y := by simp only [map_add, coe_add]
                     /-
                       🎉 no goals
                     -/
                      /-
                        S : Type u_1
                        T : Type u_2
                        R : Type u_3
                        c₁ c₂ r x✝ y : R
                        a b : QuaternionAlgebra R c₁ c₂
                        inst✝² : CommRing R
                        inst✝¹ : CommSemiring S
                        inst✝ : Algebra S R
                        s : S
                        x : QuaternionAlgebra R c₁ c₂
                        ⊢ Eq (HSMul.hSMul s x) (HMul.hMul ({ toFun := fun s => ↑((algebraMap S R) s),  …
                      -/
                      /-
                        S : Type u_1
                        T : Type u_2
                        R : Type u_3
                        c₁ c₂ r x✝ y : R
                        a b : QuaternionAlgebra R c₁ c₂
                        inst✝² : CommRing R
                        inst✝¹ : CommSemiring S
                        inst✝ : Algebra S R
                        s : S
                        x : QuaternionAlgebra R c₁ c₂
                        ⊢ Eq (HMul.hMul ({ toFun := fun s => ↑((algebraMap S R) s), map_one' := ⋯, map …
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
  smul_def' s x := by ext <;> simp [Algebra.smul_def]
                              /-
                                🎉 no goals
                              -/
  commutes' s x := by ext <;> simp [Algebra.commutes]


theorem algebraMap_eq (r : R) : algebraMap R ℍ[R,c₁,c₂] r = ⟨r, 0, 0, 0⟩ :=
  rfl


theorem algebraMap_injective : (algebraMap R ℍ[R,c₁,c₂] : _ → _).Injective :=
               /-
                 R : Type u_3
                 c₁ c₂ : R
                 inst✝ : CommRing R
                 x✝¹ x✝ : R
                 ⊢ Eq ((algebraMap R (QuaternionAlgebra R c₁ c₂)) x✝¹) ((algebraMap R (Quaterni …
               -/
  fun _ _ ↦ by simp [algebraMap_eq]
               /-
                 🎉 no goals
               -/


instance [NoZeroDivisors R] : NoZeroSMulDivisors R ℍ[R,c₁,c₂] := ⟨by
  /-
    S : Type u_1
    T : Type u_2
    R : Type u_3
    c₁ c₂ r x y : R
    a b : QuaternionAlgebra R c₁ c₂
    inst✝¹ : CommRing R
    inst✝ : NoZeroDivisors R
    ⊢ ∀ {c : R} {x : QuaternionAlgebra R c₁ c₂}, Eq (HSMul.hSMul c x) 0 → Or (Eq c …
  -/
  rintro t ⟨a, b, c, d⟩ h
  /-
    case mk
    S : Type u_1
    T : Type u_2
    R : Type u_3
    c₁ c₂ r x y : R
    a✝ b✝ : QuaternionAlgebra R c₁ c₂
    inst✝¹ : CommRing R
    inst✝ : NoZeroDivisors R
    t a b c d : R
    h : Eq (HSMul.hSMul t { re := a, imI := b, imJ := c, imK := d }) 0
    ⊢ Or (Eq t 0) (Eq { re := a, imI := b, imJ := c, imK := d } 0)
  -/
  rw [or_iff_not_imp_left]
  /-
    case mk
    S : Type u_1
    T : Type u_2
    R : Type u_3
    c₁ c₂ r x y : R
    a✝ b✝ : QuaternionAlgebra R c₁ c₂
    inst✝¹ : CommRing R
    inst✝ : NoZeroDivisors R
    t a b c d : R
    h : Eq (HSMul.hSMul t { re := a, imI := b, imJ := c, imK := d }) 0
    ⊢ Not (Eq t 0) → Eq { re := a, imI := b, imJ := c, imK := d } 0
  -/
  intro ht
  /-
    case mk
    S : Type u_1
    T : Type u_2
    R : Type u_3
    c₁ c₂ r x y : R
    a✝ b✝ : QuaternionAlgebra R c₁ c₂
    inst✝¹ : CommRing R
    inst✝ : NoZeroDivisors R
    t a b c d : R
    h : Eq (HSMul.hSMul t { re := a, imI := b, imJ := c, imK := d }) 0
    ht : Not (Eq t 0)
    ⊢ Eq { re := a, imI := b, imJ := c, imK := d } 0
  -/
  simpa [QuaternionAlgebra.ext_iff, ht] using h⟩
  /-
    🎉 no goals
  -/


/-- `QuaternionAlgebra.re` as a `LinearMap`-/
@[simps]
def reₗ : ℍ[R,c₁,c₂] →ₗ[R] R where
  toFun := re
  map_add' _ _ := rfl
  map_smul' _ _ := rfl


/-- `QuaternionAlgebra.imI` as a `LinearMap`-/
@[simps]
def imIₗ : ℍ[R,c₁,c₂] →ₗ[R] R where
  toFun := imI
  map_add' _ _ := rfl
  map_smul' _ _ := rfl


/-- `QuaternionAlgebra.imJ` as a `LinearMap`-/
@[simps]
def imJₗ : ℍ[R,c₁,c₂] →ₗ[R] R where
  toFun := imJ
  map_add' _ _ := rfl
  map_smul' _ _ := rfl


/-- `QuaternionAlgebra.imK` as a `LinearMap`-/
@[simps]
def imKₗ : ℍ[R,c₁,c₂] →ₗ[R] R where
  toFun := imK
  map_add' _ _ := rfl
  map_smul' _ _ := rfl


/-- `QuaternionAlgebra.equivTuple` as a linear equivalence. -/
def linearEquivTuple : ℍ[R,c₁,c₂] ≃ₗ[R] Fin 4 → R :=
  LinearEquiv.symm -- proofs are not `rfl` in the forward direction
    { (equivTuple c₁ c₂).symm with
      toFun := (equivTuple c₁ c₂).symm
      invFun := equivTuple c₁ c₂
      map_add' := fun _ _ => rfl
      map_smul' := fun _ _ => rfl }


@[simp]
theorem coe_linearEquivTuple : ⇑(linearEquivTuple c₁ c₂) = equivTuple c₁ c₂ :=
  rfl


@[simp]
theorem coe_linearEquivTuple_symm : ⇑(linearEquivTuple c₁ c₂).symm = (equivTuple c₁ c₂).symm :=
  rfl


/-- `ℍ[R, c₁, c₂]` has a basis over `R` given by `1`, `i`, `j`, and `k`. -/
noncomputable def basisOneIJK : Basis (Fin 4) R ℍ[R,c₁,c₂] :=
  .ofEquivFun <| linearEquivTuple c₁ c₂


@[simp]
theorem coe_basisOneIJK_repr (q : ℍ[R,c₁,c₂]) :
    ⇑((basisOneIJK c₁ c₂).repr q) = ![q.re, q.imI, q.imJ, q.imK] :=
  rfl


instance : Module.Finite R ℍ[R,c₁,c₂] := .of_basis (basisOneIJK c₁ c₂)


instance : Module.Free R ℍ[R,c₁,c₂] := .of_basis (basisOneIJK c₁ c₂)


theorem rank_eq_four [StrongRankCondition R] : Module.rank R ℍ[R,c₁,c₂] = 4 := by
  /-
    R : Type u_3
    c₁ c₂ : R
    inst✝¹ : CommRing R
    inst✝ : StrongRankCondition R
    ⊢ Eq (Module.rank R (QuaternionAlgebra R c₁ c₂)) 4
  -/
  rw [rank_eq_card_basis (basisOneIJK c₁ c₂), Fintype.card_fin]
  /-
    R : Type u_3
    c₁ c₂ : R
    inst✝¹ : CommRing R
    inst✝ : StrongRankCondition R
    ⊢ Eq (↑4) 4
  -/
  norm_num
  /-
    🎉 no goals
  -/


theorem finrank_eq_four [StrongRankCondition R] : Module.finrank R ℍ[R,c₁,c₂] = 4 := by
  /-
    R : Type u_3
    c₁ c₂ : R
    inst✝¹ : CommRing R
    inst✝ : StrongRankCondition R
    ⊢ Eq (Module.finrank R (QuaternionAlgebra R c₁ c₂)) 4
  -/
  rw [Module.finrank, rank_eq_four, Cardinal.toNat_ofNat]
  /-
    🎉 no goals
  -/


/-- There is a natural equivalence when swapping the coefficients of a quaternion algebra. -/
@[simps]
def swapEquiv : ℍ[R,c₁,c₂] ≃ₐ[R] ℍ[R, c₂, c₁] where
  toFun t := ⟨t.1, t.3, t.2, -t.4⟩
  invFun t := ⟨t.1, t.3, t.2, -t.4⟩
                   /-
                     S : Type u_1
                     T : Type u_2
                     R : Type u_3
                     c₁ c₂ r x y : R
                     a b : QuaternionAlgebra R c₁ c₂
                     inst✝ : CommRing R
                     x✝ : QuaternionAlgebra R c₁ c₂
                     ⊢ Eq ((fun t => { re := t.re, imI := t.imJ, imJ := t.imI, imK := Neg.neg t.imK …
                   -/
  left_inv _ := by simp
                   /-
                     🎉 no goals
                   -/
                    /-
                      S : Type u_1
                      T : Type u_2
                      R : Type u_3
                      c₁ c₂ r x y : R
                      a b : QuaternionAlgebra R c₁ c₂
                      inst✝ : CommRing R
                      x✝ : QuaternionAlgebra R c₂ c₁
                      ⊢ Eq ((fun t => { re := t.re, imI := t.imJ, imJ := t.imI, imK := Neg.neg t.imK …
                    -/
  right_inv _ := by simp
                    /-
                      🎉 no goals
                    -/
  map_mul' _ _ := by
    /-
      S : Type u_1
      T : Type u_2
      R : Type u_3
      c₁ c₂ r x y : R
      a b : QuaternionAlgebra R c₁ c₂
      inst✝ : CommRing R
      x✝¹ x✝ : QuaternionAlgebra R c₁ c₂
      ⊢ Eq ({ toFun := fun t => { re := t.re, imI := t.imJ, imJ := t.imI, imK := Neg …
    -/
    ext
      <;> simp only [mul_re, mul_imJ, mul_imI, add_left_inj, mul_imK, neg_mul, neg_add_rev,
                     neg_sub, mk_mul_mk, mul_neg, neg_neg, sub_neg_eq_add]
          /-
            case re
            S : Type u_1
            T : Type u_2
            R : Type u_3
            c₁ c₂ r x y : R
            a b : QuaternionAlgebra R c₁ c₂
            inst✝ : CommRing R
            x✝¹ x✝ : QuaternionAlgebra R c₁ c₂
            ⊢ Eq (HSub.hSub (HAdd.hAdd (HAdd.hAdd (HMul.hMul x✝¹.re x✝.re) (HMul.hMul (HMu …
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
      <;> ring
          /-
            🎉 no goals
          -/
                     /-
                       S : Type u_1
                       T : Type u_2
                       R : Type u_3
                       c₁ c₂ r x y : R
                       a b : QuaternionAlgebra R c₁ c₂
                       inst✝ : CommRing R
                       x✝¹ x✝ : QuaternionAlgebra R c₁ c₂
                       ⊢ Eq ({ toFun := fun t => { re := t.re, imI := t.imJ, imJ := t.imI, imK := Neg …
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
  map_add' _ _ := by ext <;> simp [add_comm]
                             /-
                               🎉 no goals
                             -/
                    /-
                      S : Type u_1
                      T : Type u_2
                      R : Type u_3
                      c₁ c₂ r x y : R
                      a b : QuaternionAlgebra R c₁ c₂
                      inst✝ : CommRing R
                      x✝ : R
                      ⊢ Eq ({ toFun := fun t => { re := t.re, imI := t.imJ, imJ := t.imI, imK := Neg …
                    -/
  commutes' _ := by simp [algebraMap_eq]
                    /-
                      🎉 no goals
                    -/


@[norm_cast, simp]
theorem coe_sub : ((x - y : R) : ℍ[R,c₁,c₂]) = x - y :=
  (algebraMap R ℍ[R,c₁,c₂]).map_sub x y


@[norm_cast, simp]
theorem coe_pow (n : ℕ) : (↑(x ^ n) : ℍ[R,c₁,c₂]) = (x : ℍ[R,c₁,c₂]) ^ n :=
  (algebraMap R ℍ[R,c₁,c₂]).map_pow x n


theorem coe_commutes : ↑r * a = a * r :=
  Algebra.commutes r a


theorem coe_commute : Commute (↑r) a :=
  coe_commutes r a


theorem coe_mul_eq_smul : ↑r * a = r • a :=
  (Algebra.smul_def r a).symm


                                              /-
                                                R : Type u_3
                                                c₁ c₂ r : R
                                                a : QuaternionAlgebra R c₁ c₂
                                                inst✝ : CommRing R
                                                ⊢ Eq (HMul.hMul a ↑r) (HSMul.hSMul r a)
                                              -/
theorem mul_coe_eq_smul : a * r = r • a := by rw [← coe_commutes, coe_mul_eq_smul]
                                              /-
                                                🎉 no goals
                                              -/


@[norm_cast, simp]
theorem coe_algebraMap : ⇑(algebraMap R ℍ[R,c₁,c₂]) = coe :=
  rfl


                                                         /-
                                                           R : Type u_3
                                                           c₁ c₂ x y : R
                                                           inst✝ : CommRing R
                                                           ⊢ Eq (HSMul.hSMul x ↑y) ↑(HMul.hMul x y)
                                                         -/
theorem smul_coe : x • (y : ℍ[R,c₁,c₂]) = ↑(x * y) := by rw [coe_mul, coe_mul_eq_smul]
                                                         /-
                                                           🎉 no goals
                                                         -/


/-- Quaternion conjugate. -/
instance instStarQuaternionAlgebra : Star ℍ[R,c₁,c₂] where star a := ⟨a.1, -a.2, -a.3, -a.4⟩


@[simp] theorem re_star : (star a).re = a.re := rfl


@[simp]
theorem imI_star : (star a).imI = -a.imI :=
  rfl


@[simp]
theorem imJ_star : (star a).imJ = -a.imJ :=
  rfl


@[simp]
theorem imK_star : (star a).imK = -a.imK :=
  rfl


@[simp]
theorem im_star : (star a).im = -a.im :=
  QuaternionAlgebra.ext neg_zero.symm rfl rfl rfl


@[simp]
theorem star_mk (a₁ a₂ a₃ a₄ : R) : star (mk a₁ a₂ a₃ a₄ : ℍ[R,c₁,c₂]) = ⟨a₁, -a₂, -a₃, -a₄⟩ :=
  rfl


instance instStarRing : StarRing ℍ[R,c₁,c₂] where
                          /-
                            S : Type u_1
                            T : Type u_2
                            R : Type u_3
                            c₁ c₂ r x✝ y : R
                            a b : QuaternionAlgebra R c₁ c₂
                            inst✝ : CommRing R
                            x : QuaternionAlgebra R c₁ c₂
                            ⊢ Eq (Star.star (Star.star x)) x
                          -/
  star_involutive x := by simp [Star.star]
                          /-
                            🎉 no goals
                          -/
                     /-
                       S : Type u_1
                       T : Type u_2
                       R : Type u_3
                       c₁ c₂ r x y : R
                       a✝ b✝ : QuaternionAlgebra R c₁ c₂
                       inst✝ : CommRing R
                       a b : QuaternionAlgebra R c₁ c₂
                       ⊢ Eq (Star.star (HAdd.hAdd a b)) (HAdd.hAdd (Star.star a) (Star.star b))
                     -/
                     /-
                       S : Type u_1
                       T : Type u_2
                       R : Type u_3
                       c₁ c₂ r x y : R
                       a✝ b✝ : QuaternionAlgebra R c₁ c₂
                       inst✝ : CommRing R
                       a b : QuaternionAlgebra R c₁ c₂
                       ⊢ Eq (Star.star (HMul.hMul a b)) (HMul.hMul (Star.star b) (Star.star a))
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
  star_add a b := by ext <;> simp [add_comm]
                             /-
                               🎉 no goals
                             -/
  star_mul a b := by ext <;> simp <;> ring


                                                        /-
                                                          R : Type u_3
                                                          c₁ c₂ : R
                                                          a : QuaternionAlgebra R c₁ c₂
                                                          inst✝ : CommRing R
                                                          ⊢ Eq (HAdd.hAdd a (Star.star a)) ↑(HMul.hMul 2 a.re)
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
theorem self_add_star' : a + star a = ↑(2 * a.re) := by ext <;> simp [two_mul]
                                                                /-
                                                                  🎉 no goals
                                                                -/


                                                    /-
                                                      R : Type u_3
                                                      c₁ c₂ : R
                                                      a : QuaternionAlgebra R c₁ c₂
                                                      inst✝ : CommRing R
                                                      ⊢ Eq (HAdd.hAdd a (Star.star a)) (HMul.hMul 2 ↑a.re)
                                                    -/
theorem self_add_star : a + star a = 2 * a.re := by simp only [self_add_star', two_mul, coe_add]
                                                    /-
                                                      🎉 no goals
                                                    -/


                                                        /-
                                                          R : Type u_3
                                                          c₁ c₂ : R
                                                          a : QuaternionAlgebra R c₁ c₂
                                                          inst✝ : CommRing R
                                                          ⊢ Eq (HAdd.hAdd (Star.star a) a) ↑(HMul.hMul 2 a.re)
                                                        -/
theorem star_add_self' : star a + a = ↑(2 * a.re) := by rw [add_comm, self_add_star']
                                                        /-
                                                          🎉 no goals
                                                        -/


                                                    /-
                                                      R : Type u_3
                                                      c₁ c₂ : R
                                                      a : QuaternionAlgebra R c₁ c₂
                                                      inst✝ : CommRing R
                                                      ⊢ Eq (HAdd.hAdd (Star.star a) a) (HMul.hMul 2 ↑a.re)
                                                    -/
theorem star_add_self : star a + a = 2 * a.re := by rw [add_comm, self_add_star]
                                                    /-
                                                      🎉 no goals
                                                    -/


theorem star_eq_two_re_sub : star a = ↑(2 * a.re) - a :=
  eq_sub_iff_add_eq.2 a.star_add_self'


instance : IsStarNormal a :=
  ⟨by
    /-
      S : Type u_1
      T : Type u_2
      R : Type u_3
      c₁ c₂ r x y : R
      a b : QuaternionAlgebra R c₁ c₂
      inst✝ : CommRing R
      ⊢ Commute (Star.star a) a
    -/
    rw [a.star_eq_two_re_sub]
    /-
      S : Type u_1
      T : Type u_2
      R : Type u_3
      c₁ c₂ r x y : R
      a b : QuaternionAlgebra R c₁ c₂
      inst✝ : CommRing R
      ⊢ Commute (HSub.hSub (↑(HMul.hMul 2 a.re)) a) a
    -/
    exact (coe_commute (2 * a.re) a).sub_left (Commute.refl a)⟩
    /-
      🎉 no goals
    -/


@[simp, norm_cast]
                                                   /-
                                                     R : Type u_3
                                                     c₁ c₂ x : R
                                                     inst✝ : CommRing R
                                                     ⊢ Eq (Star.star ↑x) ↑x
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
theorem star_coe : star (x : ℍ[R,c₁,c₂]) = x := by ext <;> simp
                                                           /-
                                                             🎉 no goals
                                                           -/


@[simp] theorem star_im : star a.im = -a.im := im_star _


@[simp]
theorem star_smul [Monoid S] [DistribMulAction S R] (s : S) (a : ℍ[R,c₁,c₂]) :
    star (s • a) = s • star a :=
  QuaternionAlgebra.ext rfl (smul_neg _ _).symm (smul_neg _ _).symm (smul_neg _ _).symm


                                                                              /-
                                                                                R : Type u_3
                                                                                c₁ c₂ : R
                                                                                inst✝ : CommRing R
                                                                                a : QuaternionAlgebra R c₁ c₂
                                                                                x : R
                                                                                h : Eq a ↑x
                                                                                ⊢ Eq a ↑a.re
                                                                              -/
theorem eq_re_of_eq_coe {a : ℍ[R,c₁,c₂]} {x : R} (h : a = x) : a = a.re := by rw [h, coe_re]
                                                                              /-
                                                                                🎉 no goals
                                                                              -/


theorem eq_re_iff_mem_range_coe {a : ℍ[R,c₁,c₂]} :
    a = a.re ↔ a ∈ Set.range (coe : R → ℍ[R,c₁,c₂]) :=
  ⟨fun h => ⟨a.re, h.symm⟩, fun ⟨_, h⟩ => eq_re_of_eq_coe h.symm⟩


@[simp]
theorem star_eq_self {c₁ c₂ : R} {a : ℍ[R,c₁,c₂]} : star a = a ↔ a = a.re := by
  /-
    R : Type u_3
    inst✝² : CommRing R
    inst✝¹ : NoZeroDivisors R
    inst✝ : CharZero R
    c₁ c₂ : R
    a : QuaternionAlgebra R c₁ c₂
    ⊢ Iff (Eq (Star.star a) a) (Eq a ↑a.re)
  -/
  simp [QuaternionAlgebra.ext_iff, neg_eq_iff_add_eq_zero, add_self_eq_zero]
  /-
    🎉 no goals
  -/


theorem star_eq_neg {c₁ c₂ : R} {a : ℍ[R,c₁,c₂]} : star a = -a ↔ a.re = 0 := by
  /-
    R : Type u_3
    inst✝² : CommRing R
    inst✝¹ : NoZeroDivisors R
    inst✝ : CharZero R
    c₁ c₂ : R
    a : QuaternionAlgebra R c₁ c₂
    ⊢ Iff (Eq (Star.star a) (Neg.neg a)) (Eq a.re 0)
  -/
  simp [QuaternionAlgebra.ext_iff, eq_neg_iff_add_eq_zero]
  /-
    🎉 no goals
  -/


                                                             /-
                                                               R : Type u_3
                                                               c₁ c₂ : R
                                                               a : QuaternionAlgebra R c₁ c₂
                                                               inst✝ : CommRing R
                                                               ⊢ Eq (HMul.hMul (Star.star a) a) ↑(HMul.hMul (Star.star a) a).re
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
theorem star_mul_eq_coe : star a * a = (star a * a).re := by ext <;> simp <;> ring
                                                                              /-
                                                                                🎉 no goals
                                                                              -/


theorem mul_star_eq_coe : a * star a = (a * star a).re := by
  /-
    R : Type u_3
    c₁ c₂ : R
    a : QuaternionAlgebra R c₁ c₂
    inst✝ : CommRing R
    ⊢ Eq (HMul.hMul a (Star.star a)) ↑(HMul.hMul a (Star.star a)).re
  -/
  rw [← star_comm_self']
  /-
    R : Type u_3
    c₁ c₂ : R
    a : QuaternionAlgebra R c₁ c₂
    inst✝ : CommRing R
    ⊢ Eq (HMul.hMul (Star.star a) a) ↑(HMul.hMul (Star.star a) a).re
  -/
  exact a.star_mul_eq_coe
  /-
    🎉 no goals
  -/


/-- Quaternion conjugate as an `AlgEquiv` to the opposite ring. -/
def starAe : ℍ[R,c₁,c₂] ≃ₐ[R] ℍ[R,c₁,c₂]ᵐᵒᵖ :=
  { starAddEquiv.trans opAddEquiv with
    toFun := op ∘ star
    invFun := star ∘ unop
                              /-
                                S : Type u_1
                                T : Type u_2
                                R : Type u_3
                                c₁ c₂ r x✝ y✝ : R
                                a b : QuaternionAlgebra R c₁ c₂
                                inst✝ : CommRing R
                                x y : QuaternionAlgebra R c₁ c₂
                                ⊢ Eq ({ toFun := Function.comp MulOpposite.op Star.star, invFun := Function.co …
                              -/
    map_mul' := fun x y => by simp
                              /-
                                🎉 no goals
                              -/
                             /-
                               S : Type u_1
                               T : Type u_2
                               R : Type u_3
                               c₁ c₂ r✝ x y : R
                               a b : QuaternionAlgebra R c₁ c₂
                               inst✝ : CommRing R
                               r : R
                               ⊢ Eq ({ toFun := Function.comp MulOpposite.op Star.star, invFun := Function.co …
                             -/
    commutes' := fun r => by simp }
                             /-
                               🎉 no goals
                             -/


@[simp]
theorem coe_starAe : ⇑(starAe : ℍ[R,c₁,c₂] ≃ₐ[R] _) = op ∘ star :=
  rfl


/-- Space of quaternions over a type. Implemented as a structure with four fields:
`re`, `im_i`, `im_j`, and `im_k`. -/
def Quaternion (R : Type*) [One R] [Neg R] :=
  QuaternionAlgebra R (-1) (-1)


@[inherit_doc]
scoped[Quaternion] notation "ℍ[" R "]" => Quaternion R


/-- The equivalence between the quaternions over `R` and `R × R × R × R`. -/
@[simps!]
def Quaternion.equivProd (R : Type*) [One R] [Neg R] : ℍ[R] ≃ R × R × R × R :=
  QuaternionAlgebra.equivProd _ _


/-- The equivalence between the quaternions over `R` and `Fin 4 → R`. -/
@[simps! symm_apply]
def Quaternion.equivTuple (R : Type*) [One R] [Neg R] : ℍ[R] ≃ (Fin 4 → R) :=
  QuaternionAlgebra.equivTuple _ _


@[simp]
theorem Quaternion.equivTuple_apply (R : Type*) [One R] [Neg R] (x : ℍ[R]) :
    Quaternion.equivTuple R x = ![x.re, x.imI, x.imJ, x.imK] :=
  rfl


instance {R : Type*} [One R] [Neg R] [Subsingleton R] : Subsingleton ℍ[R] :=
  inferInstanceAs (Subsingleton <| ℍ[R, -1, -1])

instance {R : Type*} [One R] [Neg R] [Nontrivial R] : Nontrivial ℍ[R] :=
  inferInstanceAs (Nontrivial <| ℍ[R, -1, -1])


/-- Coercion `R → ℍ[R]`. -/
@[coe] def coe : R → ℍ[R] := QuaternionAlgebra.coe


instance : CoeTC R ℍ[R] := ⟨coe⟩


instance instRing : Ring ℍ[R] := QuaternionAlgebra.instRing


instance : Inhabited ℍ[R] := inferInstanceAs <| Inhabited ℍ[R,-1,-1]


instance [SMul S R] : SMul S ℍ[R] := inferInstanceAs <| SMul S ℍ[R,-1,-1]


instance [SMul S T] [SMul S R] [SMul T R] [IsScalarTower S T R] : IsScalarTower S T ℍ[R] :=
  inferInstanceAs <| IsScalarTower S T ℍ[R,-1,-1]


instance [SMul S R] [SMul T R] [SMulCommClass S T R] : SMulCommClass S T ℍ[R] :=
  inferInstanceAs <| SMulCommClass S T ℍ[R,-1,-1]


protected instance algebra [CommSemiring S] [Algebra S R] : Algebra S ℍ[R] :=
  inferInstanceAs <| Algebra S ℍ[R,-1,-1]

-- Porting note: added shortcut

instance : Star ℍ[R] := QuaternionAlgebra.instStarQuaternionAlgebra

instance : StarRing ℍ[R] := QuaternionAlgebra.instStarRing

instance : IsStarNormal a := inferInstanceAs <| IsStarNormal (R := ℍ[R,-1,-1]) a


@[ext]
theorem ext : a.re = b.re → a.imI = b.imI → a.imJ = b.imJ → a.imK = b.imK → a = b :=
  QuaternionAlgebra.ext


/-- The imaginary part of a quaternion. -/
nonrec def im (x : ℍ[R]) : ℍ[R] := x.im


@[simp] theorem im_re : a.im.re = 0 := rfl


@[simp] theorem im_imI : a.im.imI = a.imI := rfl


@[simp] theorem im_imJ : a.im.imJ = a.imJ := rfl


@[simp] theorem im_imK : a.im.imK = a.imK := rfl


@[simp] theorem im_idem : a.im.im = a.im := rfl


@[simp] nonrec theorem re_add_im : ↑a.re + a.im = a := a.re_add_im


@[simp] nonrec theorem sub_self_im : a - a.im = a.re := a.sub_self_im


@[simp] nonrec theorem sub_self_re : a - ↑a.re = a.im := a.sub_self_re


@[simp, norm_cast]
theorem coe_re : (x : ℍ[R]).re = x := rfl


@[simp, norm_cast]
theorem coe_imI : (x : ℍ[R]).imI = 0 := rfl


@[simp, norm_cast]
theorem coe_imJ : (x : ℍ[R]).imJ = 0 := rfl


@[simp, norm_cast]
theorem coe_imK : (x : ℍ[R]).imK = 0 := rfl


@[simp, norm_cast]
theorem coe_im : (x : ℍ[R]).im = 0 := rfl


@[simp] theorem zero_re : (0 : ℍ[R]).re = 0 := rfl


@[simp] theorem zero_imI : (0 : ℍ[R]).imI = 0 := rfl


@[simp] theorem zero_imJ : (0 : ℍ[R]).imJ = 0 := rfl


@[simp] theorem zero_imK : (0 : ℍ[R]).imK = 0 := rfl


@[simp] theorem zero_im : (0 : ℍ[R]).im = 0 := rfl


@[simp, norm_cast]
theorem coe_zero : ((0 : R) : ℍ[R]) = 0 := rfl


@[simp] theorem one_re : (1 : ℍ[R]).re = 1 := rfl


@[simp] theorem one_imI : (1 : ℍ[R]).imI = 0 := rfl


@[simp] theorem one_imJ : (1 : ℍ[R]).imJ = 0 := rfl


@[simp] theorem one_imK : (1 : ℍ[R]).imK = 0 := rfl


@[simp] theorem one_im : (1 : ℍ[R]).im = 0 := rfl


@[simp, norm_cast]
theorem coe_one : ((1 : R) : ℍ[R]) = 1 := rfl


@[simp] nonrec theorem add_im : (a + b).im = a.im + b.im := a.add_im b


@[simp, norm_cast]
theorem coe_add : ((x + y : R) : ℍ[R]) = x + y :=
  QuaternionAlgebra.coe_add x y


@[simp] nonrec theorem neg_im : (-a).im = -a.im := a.neg_im


@[simp, norm_cast]
theorem coe_neg : ((-x : R) : ℍ[R]) = -x :=
  QuaternionAlgebra.coe_neg x


@[simp] nonrec theorem sub_im : (a - b).im = a.im - b.im := a.sub_im b


@[simp, norm_cast]
theorem coe_sub : ((x - y : R) : ℍ[R]) = x - y :=
  QuaternionAlgebra.coe_sub x y


@[simp]
theorem mul_re : (a * b).re = a.re * b.re - a.imI * b.imI - a.imJ * b.imJ - a.imK * b.imK :=
                                             /-
                                               R : Type u_3
                                               inst✝ : CommRing R
                                               a b : Quaternion R
                                               ⊢ Eq (HSub.hSub (HAdd.hAdd (HAdd.hAdd (HMul.hMul a.re b.re) (HMul.hMul (HMul.h …
                                             -/
  (QuaternionAlgebra.mul_re a b).trans <| by simp only [one_mul, neg_mul, sub_eq_add_neg, neg_neg]
                                             /-
                                               🎉 no goals
                                             -/


@[simp]
theorem mul_imI : (a * b).imI = a.re * b.imI + a.imI * b.re + a.imJ * b.imK - a.imK * b.imJ :=
                                              /-
                                                R : Type u_3
                                                inst✝ : CommRing R
                                                a b : Quaternion R
                                                ⊢ Eq (HAdd.hAdd (HSub.hSub (HAdd.hAdd (HMul.hMul a.re b.imI) (HMul.hMul a.imI  …
                                              -/
  (QuaternionAlgebra.mul_imI a b).trans <| by simp only [one_mul, neg_mul, sub_eq_add_neg, neg_neg]
                                              /-
                                                🎉 no goals
                                              -/


@[simp]
theorem mul_imJ : (a * b).imJ = a.re * b.imJ - a.imI * b.imK + a.imJ * b.re + a.imK * b.imI :=
                                              /-
                                                R : Type u_3
                                                inst✝ : CommRing R
                                                a b : Quaternion R
                                                ⊢ Eq (HSub.hSub (HAdd.hAdd (HAdd.hAdd (HMul.hMul a.re b.imJ) (HMul.hMul (HMul. …
                                              -/
  (QuaternionAlgebra.mul_imJ a b).trans <| by simp only [one_mul, neg_mul, sub_eq_add_neg, neg_neg]
                                              /-
                                                🎉 no goals
                                              -/


@[simp]
theorem mul_imK : (a * b).imK = a.re * b.imK + a.imI * b.imJ - a.imJ * b.imI + a.imK * b.re :=
                                              /-
                                                R : Type u_3
                                                inst✝ : CommRing R
                                                a b : Quaternion R
                                                ⊢ Eq (HAdd.hAdd (HSub.hSub (HAdd.hAdd (HMul.hMul a.re b.imK) (HMul.hMul a.imI  …
                                              -/
  (QuaternionAlgebra.mul_imK a b).trans <| by simp only [one_mul, neg_mul, sub_eq_add_neg, neg_neg]
                                              /-
                                                🎉 no goals
                                              -/


@[simp, norm_cast]
theorem coe_mul : ((x * y : R) : ℍ[R]) = x * y := QuaternionAlgebra.coe_mul x y


@[norm_cast, simp]
theorem coe_pow (n : ℕ) : (↑(x ^ n) : ℍ[R]) = (x : ℍ[R]) ^ n :=
  QuaternionAlgebra.coe_pow x n


@[simp, norm_cast]
theorem natCast_re (n : ℕ) : (n : ℍ[R]).re = n := rfl


@[simp, norm_cast]
theorem natCast_imI (n : ℕ) : (n : ℍ[R]).imI = 0 := rfl


@[simp, norm_cast]
theorem natCast_imJ (n : ℕ) : (n : ℍ[R]).imJ = 0 := rfl


@[simp, norm_cast]
theorem natCast_imK (n : ℕ) : (n : ℍ[R]).imK = 0 := rfl


@[simp, norm_cast]
theorem natCast_im (n : ℕ) : (n : ℍ[R]).im = 0 := rfl


@[norm_cast]
theorem coe_natCast (n : ℕ) : ↑(n : R) = (n : ℍ[R]) := rfl


@[simp, norm_cast]
theorem intCast_re (z : ℤ) : (z : ℍ[R]).re = z := rfl


@[simp, norm_cast]
theorem intCast_imI (z : ℤ) : (z : ℍ[R]).imI = 0 := rfl


@[simp, norm_cast]
theorem intCast_imJ (z : ℤ) : (z : ℍ[R]).imJ = 0 := rfl


@[simp, norm_cast]
theorem intCast_imK (z : ℤ) : (z : ℍ[R]).imK = 0 := rfl


@[simp, norm_cast]
theorem intCast_im (z : ℤ) : (z : ℍ[R]).im = 0 := rfl


@[norm_cast]
theorem coe_intCast (z : ℤ) : ↑(z : R) = (z : ℍ[R]) := rfl


theorem coe_injective : Function.Injective (coe : R → ℍ[R]) :=
  QuaternionAlgebra.coe_injective


@[simp]
theorem coe_inj {x y : R} : (x : ℍ[R]) = y ↔ x = y :=
  coe_injective.eq_iff


@[simp]
theorem smul_re [SMul S R] (s : S) : (s • a).re = s • a.re :=
  rfl


@[simp] theorem smul_imI [SMul S R] (s : S) : (s • a).imI = s • a.imI := rfl


@[simp] theorem smul_imJ [SMul S R] (s : S) : (s • a).imJ = s • a.imJ := rfl


@[simp] theorem smul_imK [SMul S R] (s : S) : (s • a).imK = s • a.imK := rfl


@[simp]
nonrec theorem smul_im [SMulZeroClass S R] (s : S) : (s • a).im = s • a.im :=
  a.smul_im s


@[simp, norm_cast]
theorem coe_smul [SMulZeroClass S R] (s : S) (r : R) : (↑(s • r) : ℍ[R]) = s • (r : ℍ[R]) :=
  QuaternionAlgebra.coe_smul _ _


theorem coe_commutes : ↑r * a = a * r :=
  QuaternionAlgebra.coe_commutes r a


theorem coe_commute : Commute (↑r) a :=
  QuaternionAlgebra.coe_commute r a


theorem coe_mul_eq_smul : ↑r * a = r • a :=
  QuaternionAlgebra.coe_mul_eq_smul r a


theorem mul_coe_eq_smul : a * r = r • a :=
  QuaternionAlgebra.mul_coe_eq_smul r a


@[simp]
theorem algebraMap_def : ⇑(algebraMap R ℍ[R]) = coe :=
  rfl


theorem algebraMap_injective : (algebraMap R ℍ[R] : _ → _).Injective :=
  QuaternionAlgebra.algebraMap_injective


theorem smul_coe : x • (y : ℍ[R]) = ↑(x * y) :=
  QuaternionAlgebra.smul_coe x y


instance : Module.Finite R ℍ[R] := inferInstanceAs <| Module.Finite R ℍ[R,-1,-1]

instance : Module.Free R ℍ[R] := inferInstanceAs <| Module.Free R ℍ[R,-1,-1]


theorem rank_eq_four [StrongRankCondition R] : Module.rank R ℍ[R] = 4 :=
  QuaternionAlgebra.rank_eq_four _ _


theorem finrank_eq_four [StrongRankCondition R] : Module.finrank R ℍ[R] = 4 :=
  QuaternionAlgebra.finrank_eq_four _ _


@[simp] theorem star_re : (star a).re = a.re := rfl


@[simp] theorem star_imI : (star a).imI = -a.imI := rfl


@[simp] theorem star_imJ : (star a).imJ = -a.imJ := rfl


@[simp] theorem star_imK : (star a).imK = -a.imK := rfl


@[simp] theorem star_im : (star a).im = -a.im := a.im_star


nonrec theorem self_add_star' : a + star a = ↑(2 * a.re) :=
  a.self_add_star'


nonrec theorem self_add_star : a + star a = 2 * a.re :=
  a.self_add_star


nonrec theorem star_add_self' : star a + a = ↑(2 * a.re) :=
  a.star_add_self'


nonrec theorem star_add_self : star a + a = 2 * a.re :=
  a.star_add_self


nonrec theorem star_eq_two_re_sub : star a = ↑(2 * a.re) - a :=
  a.star_eq_two_re_sub


@[simp, norm_cast]
theorem star_coe : star (x : ℍ[R]) = x :=
  QuaternionAlgebra.star_coe x


@[simp]
theorem im_star : star a.im = -a.im :=
  QuaternionAlgebra.im_star _


@[simp]
theorem star_smul [Monoid S] [DistribMulAction S R] (s : S) (a : ℍ[R]) :
    star (s • a) = s • star a :=
  QuaternionAlgebra.star_smul _ _


theorem eq_re_of_eq_coe {a : ℍ[R]} {x : R} (h : a = x) : a = a.re :=
  QuaternionAlgebra.eq_re_of_eq_coe h


theorem eq_re_iff_mem_range_coe {a : ℍ[R]} : a = a.re ↔ a ∈ Set.range (coe : R → ℍ[R]) :=
  QuaternionAlgebra.eq_re_iff_mem_range_coe


@[simp]
theorem star_eq_self {a : ℍ[R]} : star a = a ↔ a = a.re :=
  QuaternionAlgebra.star_eq_self


@[simp]
theorem star_eq_neg {a : ℍ[R]} : star a = -a ↔ a.re = 0 :=
  QuaternionAlgebra.star_eq_neg


nonrec theorem star_mul_eq_coe : star a * a = (star a * a).re :=
  a.star_mul_eq_coe


nonrec theorem mul_star_eq_coe : a * star a = (a * star a).re :=
  a.mul_star_eq_coe


/-- Quaternion conjugate as an `AlgEquiv` to the opposite ring. -/
def starAe : ℍ[R] ≃ₐ[R] ℍ[R]ᵐᵒᵖ :=
  QuaternionAlgebra.starAe


@[simp]
theorem coe_starAe : ⇑(starAe : ℍ[R] ≃ₐ[R] ℍ[R]ᵐᵒᵖ) = op ∘ star :=
  rfl


/-- Square of the norm. -/
def normSq : ℍ[R] →*₀ R where
  toFun a := (a * star a).re
                  /-
                    S : Type u_1
                    T : Type u_2
                    R : Type u_3
                    inst✝ : CommRing R
                    r x y : R
                    a b : Quaternion R
                    ⊢ Eq ((fun a => (HMul.hMul a (Star.star a)).re) 0) 0
                  -/
  map_zero' := by simp only [star_zero, zero_mul, zero_re]
                  /-
                    🎉 no goals
                  -/
                 /-
                   S : Type u_1
                   T : Type u_2
                   R : Type u_3
                   inst✝ : CommRing R
                   r x y : R
                   a b : Quaternion R
                   ⊢ Eq ({ toFun := fun a => (HMul.hMul a (Star.star a)).re, map_zero' := ⋯ }.toF …
                 -/
  map_one' := by simp only [star_one, one_mul, one_re]
                 /-
                   🎉 no goals
                 -/
  map_mul' x y := coe_injective <| by
    conv_lhs => rw [← mul_star_eq_coe, star_mul, mul_assoc, ← mul_assoc y, y.mul_star_eq_coe,
      coe_commutes, ← mul_assoc, x.mul_star_eq_coe, ← coe_mul]


theorem normSq_def : normSq a = (a * star a).re := rfl


theorem normSq_def' : normSq a = a.1 ^ 2 + a.2 ^ 2 + a.3 ^ 2 + a.4 ^ 2 := by
  simp only [normSq_def, sq, mul_neg, sub_neg_eq_add, mul_re, star_re, star_imI, star_imJ,
    star_imK]


theorem normSq_coe : normSq (x : ℍ[R]) = x ^ 2 := by
  /-
    R : Type u_3
    inst✝ : CommRing R
    x : R
    ⊢ Eq (Quaternion.normSq ↑x) (HPow.hPow x 2)
  -/
  rw [normSq_def, star_coe, ← coe_mul, coe_re, sq]
  /-
    🎉 no goals
  -/


@[simp]
                                                       /-
                                                         R : Type u_3
                                                         inst✝ : CommRing R
                                                         a : Quaternion R
                                                         ⊢ Eq (Quaternion.normSq (Star.star a)) (Quaternion.normSq a)
                                                       -/
theorem normSq_star : normSq (star a) = normSq a := by simp [normSq_def']
                                                       /-
                                                         🎉 no goals
                                                       -/


@[norm_cast]
theorem normSq_natCast (n : ℕ) : normSq (n : ℍ[R]) = (n : R) ^ 2 := by
  /-
    R : Type u_3
    inst✝ : CommRing R
    n : Nat
    ⊢ Eq (Quaternion.normSq ↑n) (HPow.hPow (↑n) 2)
  -/
  rw [← coe_natCast, normSq_coe]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-04-17")]
alias normSq_nat_cast := normSq_natCast


@[norm_cast]
theorem normSq_intCast (z : ℤ) : normSq (z : ℍ[R]) = (z : R) ^ 2 := by
  /-
    R : Type u_3
    inst✝ : CommRing R
    z : Int
    ⊢ Eq (Quaternion.normSq ↑z) (HPow.hPow (↑z) 2)
  -/
  rw [← coe_intCast, normSq_coe]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-04-17")]
alias normSq_int_cast := normSq_intCast


@[simp]
                                                  /-
                                                    R : Type u_3
                                                    inst✝ : CommRing R
                                                    a : Quaternion R
                                                    ⊢ Eq (Quaternion.normSq (Neg.neg a)) (Quaternion.normSq a)
                                                  -/
theorem normSq_neg : normSq (-a) = normSq a := by simp only [normSq_def, star_neg, neg_mul_neg]
                                                  /-
                                                    🎉 no goals
                                                  -/


                                                    /-
                                                      R : Type u_3
                                                      inst✝ : CommRing R
                                                      a : Quaternion R
                                                      ⊢ Eq (HMul.hMul a (Star.star a)) ↑(Quaternion.normSq a)
                                                    -/
theorem self_mul_star : a * star a = normSq a := by rw [mul_star_eq_coe, normSq_def]
                                                    /-
                                                      🎉 no goals
                                                    -/


                                                    /-
                                                      R : Type u_3
                                                      inst✝ : CommRing R
                                                      a : Quaternion R
                                                      ⊢ Eq (HMul.hMul (Star.star a) a) ↑(Quaternion.normSq a)
                                                    -/
theorem star_mul_self : star a * a = normSq a := by rw [star_comm_self, self_mul_star]
                                                    /-
                                                      🎉 no goals
                                                    -/


theorem im_sq : a.im ^ 2 = -normSq a.im := by
  /-
    R : Type u_3
    inst✝ : CommRing R
    a : Quaternion R
    ⊢ Eq (HPow.hPow a.im 2) (Neg.neg ↑(Quaternion.normSq a.im))
  -/
  simp_rw [sq, ← star_mul_self, im_star, neg_mul, neg_neg]
  /-
    🎉 no goals
  -/


theorem coe_normSq_add : normSq (a + b) = normSq a + a * star b + b * star a + normSq b := by
  /-
    R : Type u_3
    inst✝ : CommRing R
    a b : Quaternion R
    ⊢ Eq (↑(Quaternion.normSq (HAdd.hAdd a b))) (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd ( …
  -/
  simp only [star_add, ← self_mul_star, mul_add, add_mul, add_assoc, add_left_comm]
  /-
    🎉 no goals
  -/


theorem normSq_smul (r : R) (q : ℍ[R]) : normSq (r • q) = r ^ 2 * normSq q := by
  /-
    R : Type u_3
    inst✝ : CommRing R
    r : R
    q : Quaternion R
    ⊢ Eq (Quaternion.normSq (HSMul.hSMul r q)) (HMul.hMul (HPow.hPow r 2) (Quatern …
  -/
  simp only [normSq_def', smul_re, smul_imI, smul_imJ, smul_imK, mul_pow, mul_add, smul_eq_mul]
  /-
    🎉 no goals
  -/


theorem normSq_add (a b : ℍ[R]) : normSq (a + b) = normSq a + normSq b + 2 * (a * star b).re :=
  calc
    normSq (a + b) = normSq a + (a * star b).re + ((b * star a).re + normSq b) := by
      /-
        R : Type u_3
        inst✝ : CommRing R
        a b : Quaternion R
        ⊢ Eq (Quaternion.normSq (HAdd.hAdd a b)) (HAdd.hAdd (HAdd.hAdd (Quaternion.nor …
      -/
      simp_rw [normSq_def, star_add, add_mul, mul_add, add_re]
      /-
        🎉 no goals
      -/
                                                                        /-
                                                                          R : Type u_3
                                                                          inst✝ : CommRing R
                                                                          a b : Quaternion R
                                                                          ⊢ Eq (HAdd.hAdd (HAdd.hAdd (Quaternion.normSq a) (HMul.hMul a (Star.star b)).r …
                                                                        -/
                                                                        /-
                                                                          🎉 no goals
                                                                        -/
    _ = normSq a + normSq b + ((a * star b).re + (b * star a).re) := by abel
                                                                        /-
                                                                          🎉 no goals
                                                                        -/
    _ = normSq a + normSq b + 2 * (a * star b).re := by
      /-
        R : Type u_3
        inst✝ : CommRing R
        a b : Quaternion R
        ⊢ Eq (HAdd.hAdd (HAdd.hAdd (Quaternion.normSq a) (Quaternion.normSq b)) (HAdd. …
      -/
      rw [← add_re, ← star_mul_star a b, self_add_star', coe_re]
      /-
        🎉 no goals
      -/


@[simp]
theorem normSq_eq_zero : normSq a = 0 ↔ a = 0 := by
  /-
    R : Type u_1
    inst✝ : LinearOrderedCommRing R
    a : Quaternion R
    ⊢ Iff (Eq (Quaternion.normSq a) 0) (Eq a 0)
  -/
  refine ⟨fun h => ?_, fun h => h.symm ▸ normSq.map_zero⟩
  rw [normSq_def', add_eq_zero_iff_of_nonneg, add_eq_zero_iff_of_nonneg, add_eq_zero_iff_of_nonneg]
    at h
    /-
      R : Type u_1
      inst✝ : LinearOrderedCommRing R
      a : Quaternion R
      h : And (And (And (Eq (HPow.hPow a.re 2) 0) (Eq (HPow.hPow a.imI 2) 0)) (Eq (H …
      ⊢ Eq a 0
    -/
  · exact ext a 0 (pow_eq_zero h.1.1.1) (pow_eq_zero h.1.1.2) (pow_eq_zero h.1.2) (pow_eq_zero h.2)
    /-
      🎉 no goals
    -/
  /-
    case ha
    R : Type u_1
    inst✝ : LinearOrderedCommRing R
    a : Quaternion R
    h : And (And (Eq (HAdd.hAdd (HPow.hPow a.re 2) (HPow.hPow a.imI 2)) 0) (Eq (HP …
    ⊢ LE.le 0 (HPow.hPow a.re 2)
  -/
  all_goals apply_rules [sq_nonneg, add_nonneg]
  /-
    🎉 no goals
  -/


theorem normSq_ne_zero : normSq a ≠ 0 ↔ a ≠ 0 := normSq_eq_zero.not


@[simp]
theorem normSq_nonneg : 0 ≤ normSq a := by
  /-
    R : Type u_1
    inst✝ : LinearOrderedCommRing R
    a : Quaternion R
    ⊢ LE.le 0 (Quaternion.normSq a)
  -/
  rw [normSq_def']
  /-
    R : Type u_1
    inst✝ : LinearOrderedCommRing R
    a : Quaternion R
    ⊢ LE.le 0 (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd (HPow.hPow a.re 2) (HPow.hPow a.imI …
  -/
  apply_rules [sq_nonneg, add_nonneg]
  /-
    🎉 no goals
  -/


@[simp]
theorem normSq_le_zero : normSq a ≤ 0 ↔ a = 0 :=
  normSq_nonneg.le_iff_eq.trans normSq_eq_zero


instance instNontrivial : Nontrivial ℍ[R] where
  exists_pair_ne := ⟨0, 1, mt (congr_arg QuaternionAlgebra.re) zero_ne_one⟩


instance : NoZeroDivisors ℍ[R] where
  eq_zero_or_eq_zero_of_mul_eq_zero {a b} hab :=
                                         /-
                                           R : Type u_1
                                           inst✝ : LinearOrderedCommRing R
                                           a✝ a b : Quaternion R
                                           hab : Eq (HMul.hMul a b) 0
                                           ⊢ Eq (HMul.hMul (Quaternion.normSq a) (Quaternion.normSq b)) 0
                                         -/
    have : normSq a * normSq b = 0 := by rwa [← map_mul, normSq_eq_zero]
                                         /-
                                           🎉 no goals
                                         -/
    (eq_zero_or_eq_zero_of_mul_eq_zero this).imp normSq_eq_zero.1 normSq_eq_zero.1


instance : IsDomain ℍ[R] := NoZeroDivisors.to_isDomain _


theorem sq_eq_normSq : a ^ 2 = normSq a ↔ a = a.re := by
  /-
    R : Type u_1
    inst✝ : LinearOrderedCommRing R
    a : Quaternion R
    ⊢ Iff (Eq (HPow.hPow a 2) ↑(Quaternion.normSq a)) (Eq a ↑a.re)
  -/
  rw [← star_eq_self, ← star_mul_self, sq, mul_eq_mul_right_iff, eq_comm]
  /-
    R : Type u_1
    inst✝ : LinearOrderedCommRing R
    a : Quaternion R
    ⊢ Iff (Or (Eq (Star.star a) a) (Eq a 0)) (Eq (Star.star a) a)
  -/
  exact or_iff_left_of_imp fun ha ↦ ha.symm ▸ star_zero _
  /-
    🎉 no goals
  -/


theorem sq_eq_neg_normSq : a ^ 2 = -normSq a ↔ a.re = 0 := by
  /-
    R : Type u_1
    inst✝ : LinearOrderedCommRing R
    a : Quaternion R
    ⊢ Iff (Eq (HPow.hPow a 2) (Neg.neg ↑(Quaternion.normSq a))) (Eq a.re 0)
  -/
  simp_rw [← star_eq_neg]
  /-
    R : Type u_1
    inst✝ : LinearOrderedCommRing R
    a : Quaternion R
    ⊢ Iff (Eq (HPow.hPow a 2) (Neg.neg ↑(Quaternion.normSq a))) (Eq (Star.star a)  …
  -/
  obtain rfl | hq0 := eq_or_ne a 0
    /-
      case inl
      R : Type u_1
      inst✝ : LinearOrderedCommRing R
      ⊢ Iff (Eq (HPow.hPow 0 2) (Neg.neg ↑(Quaternion.normSq 0))) (Eq (Star.star 0)  …
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case inr
      R : Type u_1
      inst✝ : LinearOrderedCommRing R
      a : Quaternion R
      hq0 : Ne a 0
      ⊢ Iff (Eq (HPow.hPow a 2) (Neg.neg ↑(Quaternion.normSq a))) (Eq (Star.star a)  …
    -/
  · rw [← star_mul_self, ← mul_neg, ← neg_sq, sq, mul_left_inj' (neg_ne_zero.mpr hq0), eq_comm]
    /-
      🎉 no goals
    -/


@[simps (config := .lemmasOnly)]
instance instInv : Inv ℍ[R] :=
  ⟨fun a => (normSq a)⁻¹ • star a⟩


instance instGroupWithZero : GroupWithZero ℍ[R] :=
  { Quaternion.instNontrivial,
        /-
          R : Type u_1
          inst✝ : LinearOrderedField R
          a b : Quaternion R
          ⊢ MonoidWithZero (Quaternion R)
        -/
    (by infer_instance : MonoidWithZero ℍ[R]) with
        /-
          🎉 no goals
        -/
    inv := Inv.inv
                   /-
                     R : Type u_1
                     inst✝ : LinearOrderedField R
                     a b : Quaternion R
                     ⊢ Eq (Inv.inv 0) 0
                   -/
    inv_zero := by rw [instInv_inv, star_zero, smul_zero]
                   /-
                     🎉 no goals
                   -/
    mul_inv_cancel := fun a ha => by
      rw [instInv_inv, Algebra.mul_smul_comm (normSq a)⁻¹ a (star a), self_mul_star, smul_coe,
        inv_mul_cancel₀ (normSq_ne_zero.2 ha), coe_one] }


@[norm_cast, simp]
theorem coe_inv (x : R) : ((x⁻¹ : R) : ℍ[R]) = (↑x)⁻¹ :=
  map_inv₀ (algebraMap R ℍ[R]) _


@[norm_cast, simp]
theorem coe_div (x y : R) : ((x / y : R) : ℍ[R]) = x / y :=
  map_div₀ (algebraMap R ℍ[R]) x y


@[norm_cast, simp]
theorem coe_zpow (x : R) (z : ℤ) : ((x ^ z : R) : ℍ[R]) = (x : ℍ[R]) ^ z :=
  map_zpow₀ (algebraMap R ℍ[R]) x z


instance instNNRatCast : NNRatCast ℍ[R] where nnratCast q := (q : R)

instance instRatCast : RatCast ℍ[R] where ratCast q := (q : R)


@[simp, norm_cast] lemma re_nnratCast (q : ℚ≥0) : (q : ℍ[R]).re = q := rfl

@[simp, norm_cast] lemma im_nnratCast (q : ℚ≥0) : (q : ℍ[R]).im = 0 := rfl

@[simp, norm_cast] lemma imI_nnratCast (q : ℚ≥0) : (q : ℍ[R]).imI = 0 := rfl

@[simp, norm_cast] lemma imJ_nnratCast (q : ℚ≥0) : (q : ℍ[R]).imJ = 0 := rfl

@[simp, norm_cast] lemma imK_nnratCast (q : ℚ≥0) : (q : ℍ[R]).imK = 0 := rfl

@[simp, norm_cast] lemma ratCast_re (q : ℚ) : (q : ℍ[R]).re = q := rfl

@[simp, norm_cast] lemma ratCast_im (q : ℚ) : (q : ℍ[R]).im = 0 := rfl

@[simp, norm_cast] lemma ratCast_imI (q : ℚ) : (q : ℍ[R]).imI = 0 := rfl

@[simp, norm_cast] lemma ratCast_imJ (q : ℚ) : (q : ℍ[R]).imJ = 0 := rfl

@[simp, norm_cast] lemma ratCast_imK (q : ℚ) : (q : ℍ[R]).imK = 0 := rfl


@[deprecated (since := "2024-04-17")]
alias rat_cast_imI := ratCast_imI


@[deprecated (since := "2024-04-17")]
alias rat_cast_imJ := ratCast_imJ


@[deprecated (since := "2024-04-17")]
alias rat_cast_imK := ratCast_imK


@[norm_cast] lemma coe_nnratCast (q : ℚ≥0) : ↑(q : R) = (q : ℍ[R]) := rfl


@[norm_cast] lemma coe_ratCast (q : ℚ) : ↑(q : R) = (q : ℍ[R]) := rfl


@[deprecated (since := "2024-04-17")]
alias coe_rat_cast := coe_ratCast


instance instDivisionRing : DivisionRing ℍ[R] where
  __ := Quaternion.instGroupWithZero
  __ := Quaternion.instRing
  nnqsmul := (· • ·)
  qsmul := (· • ·)
                        /-
                          R : Type u_1
                          inst✝ : LinearOrderedField R
                          a b : Quaternion R
                          q : NNRat
                          ⊢ Eq (↑q) (HDiv.hDiv ↑q.num ↑q.den)
                        -/
  nnratCast_def q := by rw [← coe_nnratCast, NNRat.cast_def, coe_div, coe_natCast, coe_natCast]
                        /-
                          🎉 no goals
                        -/
                      /-
                        R : Type u_1
                        inst✝ : LinearOrderedField R
                        a b : Quaternion R
                        q : Rat
                        ⊢ Eq (↑q) (HDiv.hDiv ↑q.num ↑q.den)
                      -/
                        /-
                          R : Type u_1
                          inst✝ : LinearOrderedField R
                          a b : Quaternion R
                          q : NNRat
                          x : Quaternion R
                          ⊢ Eq ((fun x1 x2 => HSMul.hSMul x1 x2) q x) (HMul.hMul (↑q) x)
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
  ratCast_def q := by rw [← coe_ratCast, Rat.cast_def, coe_div, coe_intCast, coe_natCast]
                                                                       /-
                                                                         🎉 no goals
                                                                       -/
                      /-
                        🎉 no goals
                      -/
  nnqsmul_def q x := by rw [← coe_nnratCast, coe_mul_eq_smul]; ext <;> exact NNRat.smul_def _ _
                      /-
                        R : Type u_1
                        inst✝ : LinearOrderedField R
                        a b : Quaternion R
                        q : Rat
                        x : Quaternion R
                        ⊢ Eq ((fun x1 x2 => HSMul.hSMul x1 x2) q x) (HMul.hMul (↑q) x)
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
  qsmul_def q x := by rw [← coe_ratCast, coe_mul_eq_smul]; ext <;> exact Rat.smul_def _ _
                                                                   /-
                                                                     🎉 no goals
                                                                   -/


theorem normSq_inv : normSq a⁻¹ = (normSq a)⁻¹ :=
  map_inv₀ normSq _


theorem normSq_div : normSq (a / b) = normSq a / normSq b :=
  map_div₀ normSq a b


theorem normSq_zpow (z : ℤ) : normSq (a ^ z) = normSq a ^ z :=
  map_zpow₀ normSq a z


@[norm_cast]
theorem normSq_ratCast (q : ℚ) : normSq (q : ℍ[R]) = (q : ℍ[R]) ^ 2 := by
  /-
    R : Type u_1
    inst✝ : LinearOrderedField R
    q : Rat
    ⊢ Eq (↑(Quaternion.normSq ↑q)) (HPow.hPow (↑q) 2)
  -/
  rw [← coe_ratCast, normSq_coe, coe_pow]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-04-17")]
alias normSq_rat_cast := normSq_ratCast


private theorem pow_four [Infinite R] : #R ^ 4 = #R :=
                                      /-
                                        R : Type u_1
                                        inst✝ : Infinite R
                                        ⊢ LE.le 1 4
                                      -/
  power_nat_eq (aleph0_le_mk R) <| by decide
                                      /-
                                        🎉 no goals
                                      -/


/-- The cardinality of a quaternion algebra, as a type. -/
theorem mk_quaternionAlgebra : #(ℍ[R,c₁,c₂]) = #R ^ 4 := by
  /-
    R : Type u_1
    c₁ c₂ : R
    ⊢ Eq (Cardinal.mk (QuaternionAlgebra R c₁ c₂)) (HPow.hPow (Cardinal.mk R) 4)
  -/
  rw [mk_congr (QuaternionAlgebra.equivProd c₁ c₂)]
  /-
    R : Type u_1
    c₁ c₂ : R
    ⊢ Eq (Cardinal.mk (Prod R (Prod R (Prod R R)))) (HPow.hPow (Cardinal.mk R) 4)
  -/
  simp only [mk_prod, lift_id]
  /-
    R : Type u_1
    c₁ c₂ : R
    ⊢ Eq (HMul.hMul (Cardinal.mk R) (HMul.hMul (Cardinal.mk R) (HMul.hMul (Cardina …
  -/
  ring
  /-
    🎉 no goals
  -/


@[simp]
theorem mk_quaternionAlgebra_of_infinite [Infinite R] : #(ℍ[R,c₁,c₂]) = #R := by
  /-
    R : Type u_1
    c₁ c₂ : R
    inst✝ : Infinite R
    ⊢ Eq (Cardinal.mk (QuaternionAlgebra R c₁ c₂)) (Cardinal.mk R)
  -/
  rw [mk_quaternionAlgebra, pow_four]
  /-
    🎉 no goals
  -/


/-- The cardinality of a quaternion algebra, as a set. -/
theorem mk_univ_quaternionAlgebra : #(Set.univ : Set ℍ[R,c₁,c₂]) = #R ^ 4 := by
  /-
    R : Type u_1
    c₁ c₂ : R
    ⊢ Eq (Cardinal.mk ↑Set.univ) (HPow.hPow (Cardinal.mk R) 4)
  -/
  rw [mk_univ, mk_quaternionAlgebra]
  /-
    🎉 no goals
  -/


theorem mk_univ_quaternionAlgebra_of_infinite [Infinite R] :
                                            /-
                                              R : Type u_1
                                              c₁ c₂ : R
                                              inst✝ : Infinite R
                                              ⊢ Eq (Cardinal.mk ↑Set.univ) (Cardinal.mk R)
                                            -/
    #(Set.univ : Set ℍ[R,c₁,c₂]) = #R := by rw [mk_univ_quaternionAlgebra, pow_four]
                                            /-
                                              🎉 no goals
                                            -/


/-- Show the quaternion ⟨w, x, y, z⟩ as a string "{ re := w, imI := x, imJ := y, imK := z }".

For the typical case of quaternions over ℝ, each component will show as a Cauchy sequence due to
the way Real numbers are represented.
-/
instance [Repr R] {a b : R} : Repr ℍ[R, a, b] where
  reprPrec q _ :=
    s!"\{ re := {repr q.re}, imI := {repr q.imI}, imJ := {repr q.imJ}, imK := {repr q.imK} }"


/-- The cardinality of the quaternions, as a type. -/
@[simp]
theorem mk_quaternion : #(ℍ[R]) = #R ^ 4 :=
  mk_quaternionAlgebra _ _

--@[simp] Porting note: LHS can be simplified to `#R^4`

theorem mk_quaternion_of_infinite [Infinite R] : #(ℍ[R]) = #R :=
  mk_quaternionAlgebra_of_infinite _ _


/-- The cardinality of the quaternions, as a set. -/
theorem mk_univ_quaternion : #(Set.univ : Set ℍ[R]) = #R ^ 4 :=
  mk_univ_quaternionAlgebra _ _

--@[simp] Porting note: LHS can be simplified to `#R^4`

theorem mk_univ_quaternion_of_infinite [Infinite R] : #(Set.univ : Set ℍ[R]) = #R :=
  mk_univ_quaternionAlgebra_of_infinite _ _


