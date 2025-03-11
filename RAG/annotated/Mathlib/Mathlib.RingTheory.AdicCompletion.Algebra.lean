@[local simp]
theorem transitionMap_ideal_mk {m n : ℕ} (hmn : m ≤ n) (x : R) :
    transitionMap I R hmn (Ideal.Quotient.mk (I ^ n • ⊤ : Ideal R) x) =
      Ideal.Quotient.mk (I ^ m • ⊤ : Ideal R) x :=
  rfl


@[local simp]
theorem transitionMap_map_one {m n : ℕ} (hmn : m ≤ n) : transitionMap I R hmn 1 = 1 :=
  rfl


@[local simp]
theorem transitionMap_map_mul {m n : ℕ} (hmn : m ≤ n) (x y : R ⧸ (I ^ n • ⊤ : Ideal R)) :
    transitionMap I R hmn (x * y) = transitionMap I R hmn x * transitionMap I R hmn y :=
  Quotient.inductionOn₂' x y (fun _ _ ↦ rfl)


@[local simp]
theorem transitionMap_map_pow {m n a : ℕ} (hmn : m ≤ n) (x : R ⧸ (I ^ n • ⊤ : Ideal R)) :
    transitionMap I R hmn (x ^ a) = transitionMap I R hmn x ^ a :=
  Quotient.inductionOn' x (fun _ ↦ rfl)


/-- `AdicCompletion.transitionMap` as an algebra homomorphism. -/
def transitionMapₐ {m n : ℕ} (hmn : m ≤ n) :
    R ⧸ (I ^ n • ⊤ : Ideal R) →ₐ[R] R ⧸ (I ^ m • ⊤ : Ideal R) :=
  AlgHom.ofLinearMap (transitionMap I R hmn) rfl (transitionMap_map_mul I hmn)


/-- `AdicCompletion I R` is an `R`-subalgebra of `∀ n, R ⧸ (I ^ n • ⊤ : Ideal R)`. -/
def subalgebra : Subalgebra R (∀ n, R ⧸ (I ^ n • ⊤ : Ideal R)) :=
                                                     /-
                                                       R : Type u_1
                                                       S : Type u_2
                                                       inst✝³ : CommRing R
                                                       inst✝² : CommRing S
                                                       I : Ideal R
                                                       M : Type u_3
                                                       inst✝¹ : AddCommGroup M
                                                       inst✝ : Module R M
                                                       m✝ n✝ : Nat
                                                       x✝ : LE.le m✝ n✝
                                                       ⊢ Eq ((AdicCompletion.transitionMap I R x✝) (1 n✝)) (1 m✝)
                                                     -/
  Submodule.toSubalgebra (submodule I R) (fun _ ↦ by simp)
                                                     /-
                                                       🎉 no goals
                                                     -/
                                /-
                                  R : Type u_1
                                  S : Type u_2
                                  inst✝³ : CommRing R
                                  inst✝² : CommRing S
                                  I : Ideal R
                                  M : Type u_3
                                  inst✝¹ : AddCommGroup M
                                  inst✝ : Module R M
                                  x y : (n : Nat) → HasQuotient.Quotient R (HSMul.hSMul (HPow.hPow I n) Top.top)
                                  hx : Membership.mem (AdicCompletion.submodule I R) x
                                  hy : Membership.mem (AdicCompletion.submodule I R) y
                                  m n : Nat
                                  hmn : LE.le m n
                                  ⊢ Eq ((AdicCompletion.transitionMap I R hmn) (HMul.hMul x y n)) (HMul.hMul x y …
                                -/
    (fun x y hx hy m n hmn ↦ by simp [hx hmn, hy hmn])
                                /-
                                  🎉 no goals
                                -/


/-- `AdicCompletion I R` is a subring of `∀ n, R ⧸ (I ^ n • ⊤ : Ideal R)`. -/
def subring : Subring (∀ n, R ⧸ (I ^ n • ⊤ : Ideal R)) :=
  Subalgebra.toSubring (subalgebra I)


instance : Mul (AdicCompletion I R) where
                                /-
                                  R : Type u_1
                                  S : Type u_2
                                  inst✝³ : CommRing R
                                  inst✝² : CommRing S
                                  I : Ideal R
                                  M : Type u_3
                                  inst✝¹ : AddCommGroup M
                                  inst✝ : Module R M
                                  x y : AdicCompletion I R
                                  ⊢ ∀ {m n : Nat} (hmn : LE.le m n), Eq ((AdicCompletion.transitionMap I R hmn)  …
                                -/
  mul x y := ⟨x.val * y.val, by simp [x.property, y.property]⟩
                                /-
                                  🎉 no goals
                                -/


instance : One (AdicCompletion I R) where
                /-
                  R : Type u_1
                  S : Type u_2
                  inst✝³ : CommRing R
                  inst✝² : CommRing S
                  I : Ideal R
                  M : Type u_3
                  inst✝¹ : AddCommGroup M
                  inst✝ : Module R M
                  ⊢ ∀ {m n : Nat} (hmn : LE.le m n), Eq ((AdicCompletion.transitionMap I R hmn)  …
                -/
  one := ⟨1, by simp⟩
                /-
                  🎉 no goals
                -/


instance : NatCast (AdicCompletion I R) where
  natCast n := ⟨n, fun _ ↦ rfl⟩


instance : IntCast (AdicCompletion I R) where
  intCast n := ⟨n, fun _ ↦ rfl⟩


instance : Pow (AdicCompletion I R) ℕ where
                                    /-
                                      R : Type u_1
                                      S : Type u_2
                                      inst✝³ : CommRing R
                                      inst✝² : CommRing S
                                      I : Ideal R
                                      M : Type u_3
                                      inst✝¹ : AddCommGroup M
                                      inst✝ : Module R M
                                      x : AdicCompletion I R
                                      n m✝ n✝ : Nat
                                      x✝ : LE.le m✝ n✝
                                      ⊢ Eq ((AdicCompletion.transitionMap I R x✝) (HPow.hPow (↑x) n n✝)) (HPow.hPow  …
                                    -/
  pow x n := ⟨x.val ^ n, fun _ ↦ by simp [x.property]⟩
                                    /-
                                      🎉 no goals
                                    -/


instance : CommRing (AdicCompletion I R) :=
  let f : AdicCompletion I R → ∀ n, R ⧸ (I ^ n • ⊤ : Ideal R) := Subtype.val
  Subtype.val_injective.commRing f rfl rfl
    (fun _ _ ↦ rfl) (fun _ _ ↦ rfl) (fun _ ↦ rfl) (fun _ _ ↦ rfl) (fun _ _ ↦ rfl)
    (fun _ _ ↦ rfl) (fun _ _ ↦ rfl) (fun _ ↦ rfl) (fun _ ↦ rfl)


instance [Algebra S R] : Algebra S (AdicCompletion I R) where
  toFun r := ⟨algebraMap S (∀ n, R ⧸ (I ^ n • ⊤ : Ideal R)) r, by
    simp [-Ideal.Quotient.mk_algebraMap,
      IsScalarTower.algebraMap_apply S R (R ⧸ (I ^ _ • ⊤ : Ideal R))]⟩
  map_one' := Subtype.ext <| map_one _
  map_mul' x y := Subtype.ext <| map_mul _ x y
  map_zero' := Subtype.ext <| map_zero _
  map_add' x y := Subtype.ext <| map_add _ x y
  commutes' r x := Subtype.ext <| Algebra.commutes' r x.val
  smul_def' r x := Subtype.ext <| Algebra.smul_def' r x.val


@[simp]
theorem val_one (n : ℕ) : (1 : AdicCompletion I R).val n = 1 :=
  rfl


@[simp]
theorem val_mul (n : ℕ) (x y : AdicCompletion I R) : (x * y).val n = x.val n * y.val n :=
  rfl


/-- The canonical algebra map from the adic completion to `R ⧸ I ^ n`.

This is `AdicCompletion.eval` postcomposed with the algebra isomorphism
`R ⧸ (I ^ n • ⊤) ≃ₐ[R] R ⧸ I ^ n`. -/
def evalₐ (n : ℕ) : AdicCompletion I R →ₐ[R] R ⧸ I ^ n :=
                                               /-
                                                 R : Type u_1
                                                 S : Type u_2
                                                 inst✝³ : CommRing R
                                                 inst✝² : CommRing S
                                                 I : Ideal R
                                                 M : Type u_3
                                                 inst✝¹ : AddCommGroup M
                                                 inst✝ : Module R M
                                                 n : Nat
                                                 ⊢ Eq (HSMul.hSMul (HPow.hPow I n) Top.top) (HPow.hPow I n)
                                               -/
  have h : (I ^ n • ⊤ : Ideal R) = I ^ n := by ext x; simp
                                                      /-
                                                        🎉 no goals
                                                      -/
  AlgHom.comp
    (Ideal.quotientEquivAlgOfEq R h)
    (AlgHom.ofLinearMap (eval I R n) rfl (fun _ _ ↦ rfl))


@[simp]
theorem evalₐ_mk (n : ℕ) (x : AdicCauchySequence I R) :
    evalₐ I n (mk I R x) = Ideal.Quotient.mk (I ^ n) (x.val n) := by
  /-
    R : Type u_1
    inst✝ : CommRing R
    I : Ideal R
    n : Nat
    x : AdicCompletion.AdicCauchySequence I R
    ⊢ Eq ((AdicCompletion.evalₐ I n) ((AdicCompletion.mk I R) x)) ((Ideal.Quotient …
  -/
  simp [evalₐ]
  /-
    🎉 no goals
  -/


/-- `AdicCauchySequence I R` is an `R`-subalgebra of `ℕ → R`. -/
def AdicCauchySequence.subalgebra : Subalgebra R (ℕ → R) :=
  Submodule.toSubalgebra (AdicCauchySequence.submodule I R)
                      /-
                        R : Type u_1
                        S : Type u_2
                        inst✝³ : CommRing R
                        inst✝² : CommRing S
                        I : Ideal R
                        M : Type u_3
                        inst✝¹ : AddCommGroup M
                        inst✝ : Module R M
                        m n : Nat
                        x✝ : LE.le m n
                        ⊢ SModEq (HSMul.hSMul (HPow.hPow I m) Top.top) (1 m) (1 n)
                      -/
    (fun {m n} _ ↦ by simp; rfl)
                            /-
                              🎉 no goals
                            -/
    (fun x y hx hy {m n} hmn ↦ by
      /-
        R : Type u_1
        S : Type u_2
        inst✝³ : CommRing R
        inst✝² : CommRing S
        I : Ideal R
        M : Type u_3
        inst✝¹ : AddCommGroup M
        inst✝ : Module R M
        x y : Nat → R
        hx : Membership.mem (AdicCompletion.AdicCauchySequence.submodule I R) x
        hy : Membership.mem (AdicCompletion.AdicCauchySequence.submodule I R) y
        m n : Nat
        hmn : LE.le m n
        ⊢ SModEq (HSMul.hSMul (HPow.hPow I m) Top.top) (HMul.hMul x y m) (HMul.hMul x  …
      -/
      simp only [Pi.mul_apply]
      /-
        R : Type u_1
        S : Type u_2
        inst✝³ : CommRing R
        inst✝² : CommRing S
        I : Ideal R
        M : Type u_3
        inst✝¹ : AddCommGroup M
        inst✝ : Module R M
        x y : Nat → R
        hx : Membership.mem (AdicCompletion.AdicCauchySequence.submodule I R) x
        hy : Membership.mem (AdicCompletion.AdicCauchySequence.submodule I R) y
        m n : Nat
        hmn : LE.le m n
        ⊢ SModEq (HSMul.hSMul (HPow.hPow I m) Top.top) (HMul.hMul (x m) (y m)) (HMul.h …
      -/
      exact SModEq.mul (hx hmn) (hy hmn))
      /-
        🎉 no goals
      -/


/-- `AdicCauchySequence I R` is a subring of `ℕ → R`. -/
def AdicCauchySequence.subring : Subring (ℕ → R) :=
  Subalgebra.toSubring (AdicCauchySequence.subalgebra I)


instance : Mul (AdicCauchySequence I R) where
  mul x y := ⟨x.val * y.val, fun hmn ↦ SModEq.mul (x.property hmn) (y.property hmn)⟩


instance : One (AdicCauchySequence I R) where
  one := ⟨1, fun _ ↦ rfl⟩


instance : NatCast (AdicCauchySequence I R) where
  natCast n := ⟨n, fun _ ↦ rfl⟩


instance : IntCast (AdicCauchySequence I R) where
  intCast n := ⟨n, fun _ ↦ rfl⟩


instance : Pow (AdicCauchySequence I R) ℕ where
  pow x n := ⟨x.val ^ n, fun hmn ↦ SModEq.pow n (x.property hmn)⟩


instance : CommRing (AdicCauchySequence I R) :=
  let f : AdicCauchySequence I R → (ℕ → R) := Subtype.val
  Subtype.val_injective.commRing f rfl rfl
    (fun _ _ ↦ rfl) (fun _ _ ↦ rfl) (fun _ ↦ rfl) (fun _ _ ↦ rfl) (fun _ _ ↦ rfl)
    (fun _ _ ↦ rfl) (fun _ _ ↦ rfl) (fun _ ↦ rfl) (fun _ ↦ rfl)


instance : Algebra R (AdicCauchySequence I R) where
  toFun r := ⟨algebraMap R (∀ _, R) r, fun _ ↦ rfl⟩
  map_one' := Subtype.ext <| map_one _
  map_mul' x y := Subtype.ext <| map_mul _ x y
  map_zero' := Subtype.ext <| map_zero _
  map_add' x y := Subtype.ext <| map_add _ x y
  commutes' r x := Subtype.ext <| Algebra.commutes' r x.val
  smul_def' r x := Subtype.ext <| Algebra.smul_def' r x.val


@[simp]
theorem one_apply (n : ℕ) : (1 : AdicCauchySequence I R) n = 1 :=
  rfl


@[simp]
theorem mul_apply (n : ℕ) (f g : AdicCauchySequence I R) : (f * g) n = f n * g n :=
  rfl


/-- The canonical algebra map from adic cauchy sequences to the adic completion. -/
@[simps!]
def mkₐ : AdicCauchySequence I R →ₐ[R] AdicCompletion I R :=
  AlgHom.ofLinearMap (mk I R) rfl (fun _ _ ↦ rfl)


@[simp]
theorem evalₐ_mkₐ (n : ℕ) (x : AdicCauchySequence I R) :
    evalₐ I n (mkₐ I x) = Ideal.Quotient.mk (I ^ n) (x.val n) := by
  /-
    R : Type u_1
    inst✝ : CommRing R
    I : Ideal R
    n : Nat
    x : AdicCompletion.AdicCauchySequence I R
    ⊢ Eq ((AdicCompletion.evalₐ I n) ((AdicCompletion.mkₐ I) x)) ((Ideal.Quotient. …
  -/
  simp [mkₐ]
  /-
    🎉 no goals
  -/


theorem Ideal.mk_eq_mk {m n : ℕ} (hmn : m ≤ n) (r : AdicCauchySequence I R) :
    Ideal.Quotient.mk (I ^ m) (r.val n) = Ideal.Quotient.mk (I ^ m) (r.val m) := by
  /-
    R : Type u_1
    inst✝ : CommRing R
    I : Ideal R
    m n : Nat
    hmn : LE.le m n
    r : AdicCompletion.AdicCauchySequence I R
    ⊢ Eq ((Ideal.Quotient.mk (HPow.hPow I m)) (↑r n)) ((Ideal.Quotient.mk (HPow.hP …
  -/
  have h : I ^ m = I ^ m • ⊤ := by simp
  /-
    R : Type u_1
    inst✝ : CommRing R
    I : Ideal R
    m n : Nat
    hmn : LE.le m n
    r : AdicCompletion.AdicCauchySequence I R
    h : Eq (HPow.hPow I m) (HSMul.hSMul (HPow.hPow I m) Top.top)
    ⊢ Eq ((Ideal.Quotient.mk (HPow.hPow I m)) (↑r n)) ((Ideal.Quotient.mk (HPow.hP …
  -/
  rw [h, ← Ideal.Quotient.mk_eq_mk, ← Ideal.Quotient.mk_eq_mk]
  /-
    R : Type u_1
    inst✝ : CommRing R
    I : Ideal R
    m n : Nat
    hmn : LE.le m n
    r : AdicCompletion.AdicCauchySequence I R
    h : Eq (HPow.hPow I m) (HSMul.hSMul (HPow.hPow I m) Top.top)
    ⊢ Eq (Submodule.Quotient.mk (↑r n)) (Submodule.Quotient.mk (↑r m))
  -/
  exact (r.property hmn).symm
  /-
    🎉 no goals
  -/


theorem smul_mk {m n : ℕ} (hmn : m ≤ n) (r : AdicCauchySequence I R)
    (x : AdicCauchySequence I M) :
    r.val n • Submodule.Quotient.mk (p := (I ^ m • ⊤ : Submodule R M)) (x.val n) =
      r.val m • Submodule.Quotient.mk (p := (I ^ m • ⊤ : Submodule R M)) (x.val m) := by
  rw [← Submodule.Quotient.mk_smul, ← Module.Quotient.mk_smul_mk,
    AdicCauchySequence.mk_eq_mk hmn, Ideal.mk_eq_mk I hmn, Module.Quotient.mk_smul_mk,
    Submodule.Quotient.mk_smul]


/-- Scalar multiplication of `R ⧸ (I • ⊤)` on `M ⧸ (I • ⊤)`. This is used in order to have
good definitional behaviour for the module instance on adic completions -/
instance : SMul (R ⧸ (I • ⊤ : Ideal R)) (M ⧸ (I • ⊤ : Submodule R M)) where
  smul r x :=
    Quotient.liftOn r (· • x) fun b₁ b₂ h ↦ by
      /-
        R : Type u_1
        S : Type u_2
        inst✝³ : CommRing R
        inst✝² : CommRing S
        I : Ideal R
        M : Type u_3
        inst✝¹ : AddCommGroup M
        inst✝ : Module R M
        r : HasQuotient.Quotient R (HSMul.hSMul I Top.top)
        x : HasQuotient.Quotient M (HSMul.hSMul I Top.top)
        b₁ b₂ : R
        h : HasEquiv.Equiv b₁ b₂
        ⊢ Eq ((fun x_1 => HSMul.hSMul x_1 x) b₁) ((fun x_1 => HSMul.hSMul x_1 x) b₂)
      -/
      refine Quotient.inductionOn' x (fun x ↦ ?_)
      have h : b₁ - b₂ ∈ (I : Submodule R R) := by
        rwa [show I = I • ⊤ by simp, ← Submodule.quotientRel_def]
      rw [← sub_eq_zero, ← sub_smul, Submodule.Quotient.mk''_eq_mk,
        ← Submodule.Quotient.mk_smul, Submodule.Quotient.mk_eq_zero]
      /-
        R : Type u_1
        S : Type u_2
        inst✝³ : CommRing R
        inst✝² : CommRing S
        I : Ideal R
        M : Type u_3
        inst✝¹ : AddCommGroup M
        inst✝ : Module R M
        r : HasQuotient.Quotient R (HSMul.hSMul I Top.top)
        x✝ : HasQuotient.Quotient M (HSMul.hSMul I Top.top)
        b₁ b₂ : R
        h✝ : HasEquiv.Equiv b₁ b₂
        x : M
        h : Membership.mem I (HSub.hSub b₁ b₂)
        ⊢ Membership.mem (HSMul.hSMul I Top.top) (HSMul.hSMul (HSub.hSub b₁ b₂) x)
      -/
      exact Submodule.smul_mem_smul h mem_top
      /-
        🎉 no goals
      -/


@[local simp]
theorem mk_smul_mk (r : R) (x : M) :
    Ideal.Quotient.mk (I • ⊤) r • Submodule.Quotient.mk (p := (I • ⊤ : Submodule R M)) x
      = r • Submodule.Quotient.mk (p := (I • ⊤ : Submodule R M)) x :=
  rfl


theorem val_smul_eq_evalₐ_smul (n : ℕ) (r : AdicCompletion I R)
    (x : M ⧸ (I ^ n • ⊤ : Submodule R M)) : r.val n • x = evalₐ I n r • x := by
  /-
    R : Type u_1
    inst✝² : CommRing R
    I : Ideal R
    M : Type u_3
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    n : Nat
    r : AdicCompletion I R
    x : HasQuotient.Quotient M (HSMul.hSMul (HPow.hPow I n) Top.top)
    ⊢ Eq (HSMul.hSMul (↑r n) x) (HSMul.hSMul ((AdicCompletion.evalₐ I n) r) x)
  -/
  apply induction_on I R r (fun r ↦ ?_)
  /-
    R : Type u_1
    inst✝² : CommRing R
    I : Ideal R
    M : Type u_3
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    n : Nat
    r✝ : AdicCompletion I R
    x : HasQuotient.Quotient M (HSMul.hSMul (HPow.hPow I n) Top.top)
    r : AdicCompletion.AdicCauchySequence I R
    ⊢ Eq (HSMul.hSMul (↑((AdicCompletion.mk I R) r) n) x) (HSMul.hSMul ((AdicCompl …
  -/
  exact Quotient.inductionOn' x (fun x ↦ rfl)
  /-
    🎉 no goals
  -/


instance : Module (R ⧸ (I • ⊤ : Ideal R)) (M ⧸ (I • ⊤ : Submodule R M)) :=
  Function.Surjective.moduleLeft (Ideal.Quotient.mk (I • ⊤ : Ideal R))
    Ideal.Quotient.mk_surjective (fun _ _ ↦ rfl)


instance : IsScalarTower R (R ⧸ (I • ⊤ : Ideal R)) (M ⧸ (I • ⊤ : Submodule R M)) where
  smul_assoc r s x := by
    /-
      R : Type u_1
      S : Type u_2
      inst✝³ : CommRing R
      inst✝² : CommRing S
      I : Ideal R
      M : Type u_3
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      r : R
      s : HasQuotient.Quotient R (HSMul.hSMul I Top.top)
      x : HasQuotient.Quotient M (HSMul.hSMul I Top.top)
      ⊢ Eq (HSMul.hSMul (HSMul.hSMul r s) x) (HSMul.hSMul r (HSMul.hSMul s x))
    -/
    refine Quotient.inductionOn' s (fun s ↦ ?_)
    /-
      R : Type u_1
      S : Type u_2
      inst✝³ : CommRing R
      inst✝² : CommRing S
      I : Ideal R
      M : Type u_3
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      r : R
      s✝ : HasQuotient.Quotient R (HSMul.hSMul I Top.top)
      x : HasQuotient.Quotient M (HSMul.hSMul I Top.top)
      s : R
      ⊢ Eq (HSMul.hSMul (HSMul.hSMul r (Quotient.mk'' s)) x) (HSMul.hSMul r (HSMul.h …
    -/
    refine Quotient.inductionOn' x (fun x ↦ ?_)
    /-
      R : Type u_1
      S : Type u_2
      inst✝³ : CommRing R
      inst✝² : CommRing S
      I : Ideal R
      M : Type u_3
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      r : R
      s✝ : HasQuotient.Quotient R (HSMul.hSMul I Top.top)
      x✝ : HasQuotient.Quotient M (HSMul.hSMul I Top.top)
      s : R
      x : M
      ⊢ Eq (HSMul.hSMul (HSMul.hSMul r (Quotient.mk'' s)) (Quotient.mk'' x)) (HSMul. …
    -/
    simp only [Submodule.Quotient.mk''_eq_mk]
    /-
      R : Type u_1
      S : Type u_2
      inst✝³ : CommRing R
      inst✝² : CommRing S
      I : Ideal R
      M : Type u_3
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      r : R
      s✝ : HasQuotient.Quotient R (HSMul.hSMul I Top.top)
      x✝ : HasQuotient.Quotient M (HSMul.hSMul I Top.top)
      s : R
      x : M
      ⊢ Eq (HSMul.hSMul (HSMul.hSMul r (Submodule.Quotient.mk s)) (Submodule.Quotien …
    -/
    rw [← Submodule.Quotient.mk_smul, Ideal.Quotient.mk_eq_mk, mk_smul_mk, smul_assoc]
    /-
      R : Type u_1
      S : Type u_2
      inst✝³ : CommRing R
      inst✝² : CommRing S
      I : Ideal R
      M : Type u_3
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      r : R
      s✝ : HasQuotient.Quotient R (HSMul.hSMul I Top.top)
      x✝ : HasQuotient.Quotient M (HSMul.hSMul I Top.top)
      s : R
      x : M
      ⊢ Eq (HSMul.hSMul r (HSMul.hSMul s (Submodule.Quotient.mk x))) (HSMul.hSMul r  …
    -/
    rfl
    /-
      🎉 no goals
    -/


instance smul : SMul (AdicCompletion I R) (AdicCompletion I M) where
  smul r x := {
    val := fun n ↦ eval I R n r • eval I M n x
    property := fun {m n} hmn ↦ by
      /-
        R : Type u_1
        S : Type u_2
        inst✝³ : CommRing R
        inst✝² : CommRing S
        I : Ideal R
        M : Type u_3
        inst✝¹ : AddCommGroup M
        inst✝ : Module R M
        r : AdicCompletion I R
        x : AdicCompletion I M
        m n : Nat
        hmn : LE.le m n
        ⊢ Eq ((AdicCompletion.transitionMap I M hmn) ((fun n => HSMul.hSMul ((AdicComp …
      -/
      apply induction_on I R r (fun r ↦ ?_)
      /-
        R : Type u_1
        S : Type u_2
        inst✝³ : CommRing R
        inst✝² : CommRing S
        I : Ideal R
        M : Type u_3
        inst✝¹ : AddCommGroup M
        inst✝ : Module R M
        r✝ : AdicCompletion I R
        x : AdicCompletion I M
        m n : Nat
        hmn : LE.le m n
        r : AdicCompletion.AdicCauchySequence I R
        ⊢ Eq ((AdicCompletion.transitionMap I M hmn) ((fun n => HSMul.hSMul ((AdicComp …
      -/
      apply induction_on I M x (fun x ↦ ?_)
      simp only [coe_eval, mk_apply_coe, mkQ_apply, Ideal.Quotient.mk_eq_mk,
        mk_smul_mk, LinearMapClass.map_smul, transitionMap_mk]
      /-
        R : Type u_1
        S : Type u_2
        inst✝³ : CommRing R
        inst✝² : CommRing S
        I : Ideal R
        M : Type u_3
        inst✝¹ : AddCommGroup M
        inst✝ : Module R M
        r✝ : AdicCompletion I R
        x✝ : AdicCompletion I M
        m n : Nat
        hmn : LE.le m n
        r : AdicCompletion.AdicCauchySequence I R
        x : AdicCompletion.AdicCauchySequence I M
        ⊢ Eq (HSMul.hSMul (↑r n) (Submodule.Quotient.mk (↑x n))) (HSMul.hSMul (↑r m) ( …
      -/
      rw [smul_mk I hmn]
      /-
        🎉 no goals
      -/
  }


@[simp]
theorem smul_eval (n : ℕ) (r : AdicCompletion I R) (x : AdicCompletion I M) :
    (r • x).val n = r.val n • x.val n :=
  rfl


/-- `AdicCompletion I M` is naturally an `AdicCompletion I R` module. -/
instance module : Module (AdicCompletion I R) (AdicCompletion I M) where
  one_smul b := by
    /-
      R : Type u_1
      S : Type u_2
      inst✝³ : CommRing R
      inst✝² : CommRing S
      I : Ideal R
      M : Type u_3
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      b : AdicCompletion I M
      ⊢ Eq (HSMul.hSMul 1 b) b
    -/
    ext n
    /-
      case h
      R : Type u_1
      S : Type u_2
      inst✝³ : CommRing R
      inst✝² : CommRing S
      I : Ideal R
      M : Type u_3
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      b : AdicCompletion I M
      n : Nat
      ⊢ Eq (↑(HSMul.hSMul 1 b) n) (↑b n)
    -/
    simp only [smul_eval, val_one, one_smul]
    /-
      🎉 no goals
    -/
  mul_smul r s x := by
    /-
      R : Type u_1
      S : Type u_2
      inst✝³ : CommRing R
      inst✝² : CommRing S
      I : Ideal R
      M : Type u_3
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      r s : AdicCompletion I R
      x : AdicCompletion I M
      ⊢ Eq (HSMul.hSMul (HMul.hMul r s) x) (HSMul.hSMul r (HSMul.hSMul s x))
    -/
    ext n
    /-
      case h
      R : Type u_1
      S : Type u_2
      inst✝³ : CommRing R
      inst✝² : CommRing S
      I : Ideal R
      M : Type u_3
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      r s : AdicCompletion I R
      x : AdicCompletion I M
      n : Nat
      ⊢ Eq (↑(HSMul.hSMul (HMul.hMul r s) x) n) (↑(HSMul.hSMul r (HSMul.hSMul s x)) n)
    -/
    simp only [smul_eval, val_mul, mul_smul]
    /-
      🎉 no goals
    -/
                    /-
                      R : Type u_1
                      S : Type u_2
                      inst✝³ : CommRing R
                      inst✝² : CommRing S
                      I : Ideal R
                      M : Type u_3
                      inst✝¹ : AddCommGroup M
                      inst✝ : Module R M
                      r : AdicCompletion I R
                      ⊢ Eq (HSMul.hSMul r 0) 0
                    -/
  smul_zero r := by ext n; simp
                           /-
                             🎉 no goals
                           -/
                       /-
                         R : Type u_1
                         S : Type u_2
                         inst✝³ : CommRing R
                         inst✝² : CommRing S
                         I : Ideal R
                         M : Type u_3
                         inst✝¹ : AddCommGroup M
                         inst✝ : Module R M
                         r : AdicCompletion I R
                         x y : AdicCompletion I M
                         ⊢ Eq (HSMul.hSMul r (HAdd.hAdd x y)) (HAdd.hAdd (HSMul.hSMul r x) (HSMul.hSMul …
                       -/
  smul_add r x y := by ext n; simp
                              /-
                                🎉 no goals
                              -/
                       /-
                         R : Type u_1
                         S : Type u_2
                         inst✝³ : CommRing R
                         inst✝² : CommRing S
                         I : Ideal R
                         M : Type u_3
                         inst✝¹ : AddCommGroup M
                         inst✝ : Module R M
                         r s : AdicCompletion I R
                         x : AdicCompletion I M
                         ⊢ Eq (HSMul.hSMul (HAdd.hAdd r s) x) (HAdd.hAdd (HSMul.hSMul r x) (HSMul.hSMul …
                       -/
  add_smul r s x := by ext n; simp [val_smul, add_smul]
                              /-
                                🎉 no goals
                              -/
                    /-
                      R : Type u_1
                      S : Type u_2
                      inst✝³ : CommRing R
                      inst✝² : CommRing S
                      I : Ideal R
                      M : Type u_3
                      inst✝¹ : AddCommGroup M
                      inst✝ : Module R M
                      x : AdicCompletion I M
                      ⊢ Eq (HSMul.hSMul 0 x) 0
                    -/
  zero_smul x := by ext n; simp
                           /-
                             🎉 no goals
                           -/


instance : IsScalarTower R (AdicCompletion I R) (AdicCompletion I M) where
  smul_assoc r s x := by
    /-
      R : Type u_1
      S : Type u_2
      inst✝³ : CommRing R
      inst✝² : CommRing S
      I : Ideal R
      M : Type u_3
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      r : R
      s : AdicCompletion I R
      x : AdicCompletion I M
      ⊢ Eq (HSMul.hSMul (HSMul.hSMul r s) x) (HSMul.hSMul r (HSMul.hSMul s x))
    -/
    ext n
    /-
      case h
      R : Type u_1
      S : Type u_2
      inst✝³ : CommRing R
      inst✝² : CommRing S
      I : Ideal R
      M : Type u_3
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      r : R
      s : AdicCompletion I R
      x : AdicCompletion I M
      n : Nat
      ⊢ Eq (↑(HSMul.hSMul (HSMul.hSMul r s) x) n) (↑(HSMul.hSMul r (HSMul.hSMul s x) …
    -/
    rw [smul_eval, val_smul_apply, val_smul_apply, smul_eval, smul_assoc]
    /-
      🎉 no goals
    -/


