/-- The type of *double centralizers*, also known as the *multiplier algebra* and denoted by
`𝓜(𝕜, A)`, of a non-unital normed algebra.

If `x : 𝓜(𝕜, A)`, then `x.fst` and `x.snd` are what is usually referred to as $L$ and $R$. -/
structure DoubleCentralizer (𝕜 : Type u) (A : Type v) [NontriviallyNormedField 𝕜]
    [NonUnitalNormedRing A] [NormedSpace 𝕜 A] [SMulCommClass 𝕜 A A] [IsScalarTower 𝕜 A A] extends
    (A →L[𝕜] A) × (A →L[𝕜] A) where
  /-- The centrality condition that the maps linear maps intertwine one another. -/
  central : ∀ x y : A, snd x * y = x * fst y


@[inherit_doc]
scoped[MultiplierAlgebra] notation "𝓜(" 𝕜 ", " A ")" => DoubleCentralizer 𝕜 A


@[ext]
lemma DoubleCentralizer.ext (𝕜 : Type u) (A : Type v) [NontriviallyNormedField 𝕜]
    [NonUnitalNormedRing A] [NormedSpace 𝕜 A] [SMulCommClass 𝕜 A A] [IsScalarTower 𝕜 A A]
    (a b : 𝓜(𝕜, A)) (h : a.toProd = b.toProd) : a = b := by
  /-
    𝕜 : Type u
    A : Type v
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NonUnitalNormedRing A
    inst✝² : NormedSpace 𝕜 A
    inst✝¹ : SMulCommClass 𝕜 A A
    inst✝ : IsScalarTower 𝕜 A A
    a b : DoubleCentralizer 𝕜 A
    h : Eq a.toProd b.toProd
    ⊢ Eq a b
  -/
  cases a
  /-
    case mk
    𝕜 : Type u
    A : Type v
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NonUnitalNormedRing A
    inst✝² : NormedSpace 𝕜 A
    inst✝¹ : SMulCommClass 𝕜 A A
    inst✝ : IsScalarTower 𝕜 A A
    b : DoubleCentralizer 𝕜 A
    toProd✝ : Prod (ContinuousLinearMap (RingHom.id 𝕜) A A) (ContinuousLinearMap ( …
    central✝ : ∀ (x y : A), Eq (HMul.hMul (toProd✝.2 x) y) (HMul.hMul x (toProd✝.1 …
    h : Eq { toProd := toProd✝, central := central✝ }.toProd b.toProd
    ⊢ Eq { toProd := toProd✝, central := central✝ } b
  -/
  cases b
  /-
    case mk.mk
    𝕜 : Type u
    A : Type v
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NonUnitalNormedRing A
    inst✝² : NormedSpace 𝕜 A
    inst✝¹ : SMulCommClass 𝕜 A A
    inst✝ : IsScalarTower 𝕜 A A
    toProd✝¹ : Prod (ContinuousLinearMap (RingHom.id 𝕜) A A) (ContinuousLinearMap  …
    central✝¹ : ∀ (x y : A), Eq (HMul.hMul (toProd✝¹.2 x) y) (HMul.hMul x (toProd✝ …
    toProd✝ : Prod (ContinuousLinearMap (RingHom.id 𝕜) A A) (ContinuousLinearMap ( …
    central✝ : ∀ (x y : A), Eq (HMul.hMul (toProd✝.2 x) y) (HMul.hMul x (toProd✝.1 …
    h : Eq { toProd := toProd✝¹, central := central✝¹ }.toProd { toProd := toProd✝ …
    ⊢ Eq { toProd := toProd✝¹, central := central✝¹ } { toProd := toProd✝, central …
  -/
  simpa using h
  /-
    🎉 no goals
  -/


theorem range_toProd :
    Set.range toProd = { lr : (A →L[𝕜] A) × (A →L[𝕜] A) | ∀ x y, lr.2 x * y = x * lr.1 y } :=
  Set.ext fun x =>
    ⟨by
      /-
        𝕜 : Type u_1
        A : Type u_2
        inst✝⁴ : NontriviallyNormedField 𝕜
        inst✝³ : NonUnitalNormedRing A
        inst✝² : NormedSpace 𝕜 A
        inst✝¹ : SMulCommClass 𝕜 A A
        inst✝ : IsScalarTower 𝕜 A A
        x : Prod (ContinuousLinearMap (RingHom.id 𝕜) A A) (ContinuousLinearMap (RingHo …
        ⊢ Membership.mem (Set.range DoubleCentralizer.toProd) x → Membership.mem (setO …
      -/
      rintro ⟨a, rfl⟩
      /-
        case intro
        𝕜 : Type u_1
        A : Type u_2
        inst✝⁴ : NontriviallyNormedField 𝕜
        inst✝³ : NonUnitalNormedRing A
        inst✝² : NormedSpace 𝕜 A
        inst✝¹ : SMulCommClass 𝕜 A A
        inst✝ : IsScalarTower 𝕜 A A
        a : DoubleCentralizer 𝕜 A
        ⊢ Membership.mem (setOf fun lr => ∀ (x y : A), Eq (HMul.hMul (lr.2 x) y) (HMul …
      -/
      exact a.central, fun hx => ⟨⟨x, hx⟩, rfl⟩⟩
      /-
        🎉 no goals
      -/


instance instAdd : Add 𝓜(𝕜, A) where
  add a b :=
    { toProd := a.toProd + b.toProd
      central := fun x y =>
        show (a.snd + b.snd) x * y = x * (a.fst + b.fst) y by
          /-
            𝕜 : Type u_1
            A : Type u_2
            inst✝⁴ : NontriviallyNormedField 𝕜
            inst✝³ : NonUnitalNormedRing A
            inst✝² : NormedSpace 𝕜 A
            inst✝¹ : SMulCommClass 𝕜 A A
            inst✝ : IsScalarTower 𝕜 A A
            a b : DoubleCentralizer 𝕜 A
            x y : A
            ⊢ Eq (HMul.hMul ((HAdd.hAdd a.toProd.2 b.toProd.2) x) y) (HMul.hMul x ((HAdd.h …
          -/
          simp only [ContinuousLinearMap.add_apply, mul_add, add_mul, central] }
          /-
            🎉 no goals
          -/


instance instZero : Zero 𝓜(𝕜, A) where
  zero :=
    { toProd := 0
      central := fun x y => (zero_mul y).trans (mul_zero x).symm }


instance instNeg : Neg 𝓜(𝕜, A) where
  neg a :=
    { toProd := -a.toProd
      central := fun x y =>
        show -a.snd x * y = x * -a.fst y by
          /-
            𝕜 : Type u_1
            A : Type u_2
            inst✝⁴ : NontriviallyNormedField 𝕜
            inst✝³ : NonUnitalNormedRing A
            inst✝² : NormedSpace 𝕜 A
            inst✝¹ : SMulCommClass 𝕜 A A
            inst✝ : IsScalarTower 𝕜 A A
            a : DoubleCentralizer 𝕜 A
            x y : A
            ⊢ Eq (HMul.hMul (Neg.neg (a.toProd.2 x)) y) (HMul.hMul x (Neg.neg (a.toProd.1  …
          -/
          simp only [ContinuousLinearMap.neg_apply, neg_mul, mul_neg, central] }
          /-
            🎉 no goals
          -/


instance instSub : Sub 𝓜(𝕜, A) where
  sub a b :=
    { toProd := a.toProd - b.toProd
      central := fun x y =>
        show (a.snd - b.snd) x * y = x * (a.fst - b.fst) y by
          /-
            𝕜 : Type u_1
            A : Type u_2
            inst✝⁴ : NontriviallyNormedField 𝕜
            inst✝³ : NonUnitalNormedRing A
            inst✝² : NormedSpace 𝕜 A
            inst✝¹ : SMulCommClass 𝕜 A A
            inst✝ : IsScalarTower 𝕜 A A
            a b : DoubleCentralizer 𝕜 A
            x y : A
            ⊢ Eq (HMul.hMul ((HSub.hSub a.toProd.2 b.toProd.2) x) y) (HMul.hMul x ((HSub.h …
          -/
          simp only [ContinuousLinearMap.sub_apply, _root_.sub_mul, _root_.mul_sub, central] }
          /-
            🎉 no goals
          -/


instance instSMul : SMul S 𝓜(𝕜, A) where
  smul s a :=
    { toProd := s • a.toProd
      central := fun x y =>
        show (s • a.snd) x * y = x * (s • a.fst) y by
          /-
            𝕜 : Type u_1
            A : Type u_2
            inst✝¹⁰ : NontriviallyNormedField 𝕜
            inst✝⁹ : NonUnitalNormedRing A
            inst✝⁸ : NormedSpace 𝕜 A
            inst✝⁷ : SMulCommClass 𝕜 A A
            inst✝⁶ : IsScalarTower 𝕜 A A
            S : Type u_3
            inst✝⁵ : Monoid S
            inst✝⁴ : DistribMulAction S A
            inst✝³ : SMulCommClass 𝕜 S A
            inst✝² : ContinuousConstSMul S A
            inst✝¹ : IsScalarTower S A A
            inst✝ : SMulCommClass S A A
            s : S
            a : DoubleCentralizer 𝕜 A
            x y : A
            ⊢ Eq (HMul.hMul ((HSMul.hSMul s a.toProd.2) x) y) (HMul.hMul x ((HSMul.hSMul s …
          -/
          simp only [ContinuousLinearMap.smul_apply, mul_smul_comm, smul_mul_assoc, central] }
          /-
            🎉 no goals
          -/


@[simp]
theorem smul_toProd (s : S) (a : 𝓜(𝕜, A)) : (s • a).toProd = s • a.toProd :=
  rfl


theorem smul_fst (s : S) (a : 𝓜(𝕜, A)) : (s • a).fst = s • a.fst :=
  rfl


theorem smul_snd (s : S) (a : 𝓜(𝕜, A)) : (s • a).snd = s • a.snd :=
  rfl


instance instIsScalarTower [SMul S T] [IsScalarTower S T A] : IsScalarTower S T 𝓜(𝕜, A) where
  smul_assoc _ _ a := ext (𝕜 := 𝕜) (A := A) _ _ <| smul_assoc _ _ a.toProd


instance instSMulCommClass [SMulCommClass S T A] : SMulCommClass S T 𝓜(𝕜, A) where
  smul_comm _ _ a := ext (𝕜 := 𝕜) (A := A) _ _ <| smul_comm _ _ a.toProd


instance instIsCentralScalar {R : Type*} [Semiring R] [Module R A] [SMulCommClass 𝕜 R A]
    [ContinuousConstSMul R A] [IsScalarTower R A A] [SMulCommClass R A A] [Module Rᵐᵒᵖ A]
    [IsCentralScalar R A] : IsCentralScalar R 𝓜(𝕜, A) where
  op_smul_eq_smul _ a := ext (𝕜 := 𝕜) (A := A) _ _ <| op_smul_eq_smul _ a.toProd


instance instOne : One 𝓜(𝕜, A) :=
  ⟨⟨1, fun _x _y => rfl⟩⟩


instance instMul : Mul 𝓜(𝕜, A) where
  mul a b :=
    { toProd := (a.fst.comp b.fst, b.snd.comp a.snd)
                                                                              /-
                                                                                𝕜 : Type u_1
                                                                                A : Type u_2
                                                                                inst✝⁴ : NontriviallyNormedField 𝕜
                                                                                inst✝³ : NonUnitalNormedRing A
                                                                                inst✝² : NormedSpace 𝕜 A
                                                                                inst✝¹ : SMulCommClass 𝕜 A A
                                                                                inst✝ : IsScalarTower 𝕜 A A
                                                                                a b : DoubleCentralizer 𝕜 A
                                                                                x y : A
                                                                                ⊢ Eq (HMul.hMul (b.toProd.2 (a.toProd.2 x)) y) (HMul.hMul x (a.toProd.1 (b.toP …
                                                                              -/
      central := fun x y => show b.snd (a.snd x) * y = x * a.fst (b.fst y) by simp only [central] }
                                                                              /-
                                                                                🎉 no goals
                                                                              -/


instance instNatCast : NatCast 𝓜(𝕜, A) where
  natCast n :=
    ⟨n, fun x y => by
      /-
        𝕜 : Type u_1
        A : Type u_2
        inst✝⁴ : NontriviallyNormedField 𝕜
        inst✝³ : NonUnitalNormedRing A
        inst✝² : NormedSpace 𝕜 A
        inst✝¹ : SMulCommClass 𝕜 A A
        inst✝ : IsScalarTower 𝕜 A A
        n : Nat
        x y : A
        ⊢ Eq (HMul.hMul ((↑n).2 x) y) (HMul.hMul x ((↑n).1 y))
      -/
      rw [Prod.snd_natCast, Prod.fst_natCast]
      /-
        𝕜 : Type u_1
        A : Type u_2
        inst✝⁴ : NontriviallyNormedField 𝕜
        inst✝³ : NonUnitalNormedRing A
        inst✝² : NormedSpace 𝕜 A
        inst✝¹ : SMulCommClass 𝕜 A A
        inst✝ : IsScalarTower 𝕜 A A
        n : Nat
        x y : A
        ⊢ Eq (HMul.hMul (↑n x) y) (HMul.hMul x (↑n y))
      -/
      simp only [← Nat.smul_one_eq_cast, smul_apply, one_apply, mul_smul_comm, smul_mul_assoc]⟩
      /-
        🎉 no goals
      -/


instance instIntCast : IntCast 𝓜(𝕜, A) where
  intCast n :=
    ⟨n, fun x y => by
      /-
        𝕜 : Type u_1
        A : Type u_2
        inst✝⁴ : NontriviallyNormedField 𝕜
        inst✝³ : NonUnitalNormedRing A
        inst✝² : NormedSpace 𝕜 A
        inst✝¹ : SMulCommClass 𝕜 A A
        inst✝ : IsScalarTower 𝕜 A A
        n : Int
        x y : A
        ⊢ Eq (HMul.hMul ((↑n).2 x) y) (HMul.hMul x ((↑n).1 y))
      -/
      rw [Prod.snd_intCast, Prod.fst_intCast]
      /-
        𝕜 : Type u_1
        A : Type u_2
        inst✝⁴ : NontriviallyNormedField 𝕜
        inst✝³ : NonUnitalNormedRing A
        inst✝² : NormedSpace 𝕜 A
        inst✝¹ : SMulCommClass 𝕜 A A
        inst✝ : IsScalarTower 𝕜 A A
        n : Int
        x y : A
        ⊢ Eq (HMul.hMul (↑n x) y) (HMul.hMul x (↑n y))
      -/
      simp only [← Int.smul_one_eq_cast, smul_apply, one_apply, mul_smul_comm, smul_mul_assoc]⟩
      /-
        🎉 no goals
      -/


instance instPow : Pow 𝓜(𝕜, A) ℕ where
  pow a n :=
    ⟨a.toProd ^ n, fun x y => by
      /-
        𝕜 : Type u_1
        A : Type u_2
        inst✝⁴ : NontriviallyNormedField 𝕜
        inst✝³ : NonUnitalNormedRing A
        inst✝² : NormedSpace 𝕜 A
        inst✝¹ : SMulCommClass 𝕜 A A
        inst✝ : IsScalarTower 𝕜 A A
        a : DoubleCentralizer 𝕜 A
        n : Nat
        x y : A
        ⊢ Eq (HMul.hMul ((HPow.hPow a.toProd n).2 x) y) (HMul.hMul x ((HPow.hPow a.toP …
      -/
      induction' n with k hk generalizing x y
        /-
          case zero
          𝕜 : Type u_1
          A : Type u_2
          inst✝⁴ : NontriviallyNormedField 𝕜
          inst✝³ : NonUnitalNormedRing A
          inst✝² : NormedSpace 𝕜 A
          inst✝¹ : SMulCommClass 𝕜 A A
          inst✝ : IsScalarTower 𝕜 A A
          a : DoubleCentralizer 𝕜 A
          x y : A
          ⊢ Eq (HMul.hMul ((HPow.hPow a.toProd 0).2 x) y) (HMul.hMul x ((HPow.hPow a.toP …
        -/
      · rfl
        /-
          🎉 no goals
        -/
        /-
          case succ
          𝕜 : Type u_1
          A : Type u_2
          inst✝⁴ : NontriviallyNormedField 𝕜
          inst✝³ : NonUnitalNormedRing A
          inst✝² : NormedSpace 𝕜 A
          inst✝¹ : SMulCommClass 𝕜 A A
          inst✝ : IsScalarTower 𝕜 A A
          a : DoubleCentralizer 𝕜 A
          k : Nat
          hk : ∀ (x y : A), Eq (HMul.hMul ((HPow.hPow a.toProd k).2 x) y) (HMul.hMul x ( …
          x y : A
          ⊢ Eq (HMul.hMul ((HPow.hPow a.toProd (HAdd.hAdd k 1)).2 x) y) (HMul.hMul x ((H …
        -/
      · rw [Prod.pow_snd, Prod.pow_fst] at hk ⊢
        /-
          case succ
          𝕜 : Type u_1
          A : Type u_2
          inst✝⁴ : NontriviallyNormedField 𝕜
          inst✝³ : NonUnitalNormedRing A
          inst✝² : NormedSpace 𝕜 A
          inst✝¹ : SMulCommClass 𝕜 A A
          inst✝ : IsScalarTower 𝕜 A A
          a : DoubleCentralizer 𝕜 A
          k : Nat
          hk : ∀ (x y : A), Eq (HMul.hMul ((HPow.hPow a.toProd.2 k) x) y) (HMul.hMul x ( …
          x y : A
          ⊢ Eq (HMul.hMul ((HPow.hPow a.toProd.2 (HAdd.hAdd k 1)) x) y) (HMul.hMul x ((H …
        -/
        rw [pow_succ' a.snd, mul_apply, a.central, hk, pow_succ a.fst, mul_apply]⟩
        /-
          🎉 no goals
        -/


instance instInhabited : Inhabited 𝓜(𝕜, A) :=
  ⟨0⟩


@[simp]
theorem add_toProd (a b : 𝓜(𝕜, A)) : (a + b).toProd = a.toProd + b.toProd :=
  rfl


@[simp]
theorem zero_toProd : (0 : 𝓜(𝕜, A)).toProd = 0 :=
  rfl


@[simp]
theorem neg_toProd (a : 𝓜(𝕜, A)) : (-a).toProd = -a.toProd :=
  rfl


@[simp]
theorem sub_toProd (a b : 𝓜(𝕜, A)) : (a - b).toProd = a.toProd - b.toProd :=
  rfl


@[simp]
theorem one_toProd : (1 : 𝓜(𝕜, A)).toProd = 1 :=
  rfl


@[simp]
theorem natCast_toProd (n : ℕ) : (n : 𝓜(𝕜, A)).toProd = n :=
  rfl


@[deprecated (since := "2024-04-17")]
alias nat_cast_toProd := natCast_toProd


@[simp]
theorem intCast_toProd (n : ℤ) : (n : 𝓜(𝕜, A)).toProd = n :=
  rfl


@[deprecated (since := "2024-04-17")]
alias int_cast_toProd := intCast_toProd


@[simp]
theorem pow_toProd (n : ℕ) (a : 𝓜(𝕜, A)) : (a ^ n).toProd = a.toProd ^ n :=
  rfl


theorem add_fst (a b : 𝓜(𝕜, A)) : (a + b).fst = a.fst + b.fst :=
  rfl


theorem add_snd (a b : 𝓜(𝕜, A)) : (a + b).snd = a.snd + b.snd :=
  rfl


theorem zero_fst : (0 : 𝓜(𝕜, A)).fst = 0 :=
  rfl


theorem zero_snd : (0 : 𝓜(𝕜, A)).snd = 0 :=
  rfl


theorem neg_fst (a : 𝓜(𝕜, A)) : (-a).fst = -a.fst :=
  rfl


theorem neg_snd (a : 𝓜(𝕜, A)) : (-a).snd = -a.snd :=
  rfl


theorem sub_fst (a b : 𝓜(𝕜, A)) : (a - b).fst = a.fst - b.fst :=
  rfl


theorem sub_snd (a b : 𝓜(𝕜, A)) : (a - b).snd = a.snd - b.snd :=
  rfl


theorem one_fst : (1 : 𝓜(𝕜, A)).fst = 1 :=
  rfl


theorem one_snd : (1 : 𝓜(𝕜, A)).snd = 1 :=
  rfl


@[simp]
theorem mul_fst (a b : 𝓜(𝕜, A)) : (a * b).fst = a.fst * b.fst :=
  rfl


@[simp]
theorem mul_snd (a b : 𝓜(𝕜, A)) : (a * b).snd = b.snd * a.snd :=
  rfl


theorem natCast_fst (n : ℕ) : (n : 𝓜(𝕜, A)).fst = n :=
  rfl


@[deprecated (since := "2024-04-17")]
alias nat_cast_fst := natCast_fst


theorem natCast_snd (n : ℕ) : (n : 𝓜(𝕜, A)).snd = n :=
  rfl


@[deprecated (since := "2024-04-17")]
alias nat_cast_snd := natCast_snd


theorem intCast_fst (n : ℤ) : (n : 𝓜(𝕜, A)).fst = n :=
  rfl


@[deprecated (since := "2024-04-17")]
alias int_cast_fst := intCast_fst


theorem intCast_snd (n : ℤ) : (n : 𝓜(𝕜, A)).snd = n :=
  rfl


@[deprecated (since := "2024-04-17")]
alias int_cast_snd := intCast_snd


theorem pow_fst (n : ℕ) (a : 𝓜(𝕜, A)) : (a ^ n).fst = a.fst ^ n :=
  rfl


theorem pow_snd (n : ℕ) (a : 𝓜(𝕜, A)) : (a ^ n).snd = a.snd ^ n :=
  rfl


/-- The natural injection from `DoubleCentralizer.toProd` except the second coordinate inherits
`MulOpposite.op`. The ring structure on `𝓜(𝕜, A)` is the pullback under this map. -/
def toProdMulOpposite : 𝓜(𝕜, A) → (A →L[𝕜] A) × (A →L[𝕜] A)ᵐᵒᵖ := fun a =>
  (a.fst, MulOpposite.op a.snd)


theorem toProdMulOpposite_injective :
    Function.Injective (toProdMulOpposite : 𝓜(𝕜, A) → (A →L[𝕜] A) × (A →L[𝕜] A)ᵐᵒᵖ) :=
  fun _a _b h =>
    let h' := Prod.ext_iff.mp h
    ext (𝕜 := 𝕜) (A := A) _ _ <| Prod.ext h'.1 <| MulOpposite.op_injective h'.2


theorem range_toProdMulOpposite :
    Set.range toProdMulOpposite =
      { lr : (A →L[𝕜] A) × (A →L[𝕜] A)ᵐᵒᵖ | ∀ x y, unop lr.2 x * y = x * lr.1 y } :=
  Set.ext fun x =>
    ⟨by
      /-
        𝕜 : Type u_1
        A : Type u_2
        inst✝⁴ : NontriviallyNormedField 𝕜
        inst✝³ : NonUnitalNormedRing A
        inst✝² : NormedSpace 𝕜 A
        inst✝¹ : SMulCommClass 𝕜 A A
        inst✝ : IsScalarTower 𝕜 A A
        x : Prod (ContinuousLinearMap (RingHom.id 𝕜) A A) (MulOpposite (ContinuousLine …
        ⊢ Membership.mem (Set.range DoubleCentralizer.toProdMulOpposite) x → Membershi …
      -/
      rintro ⟨a, rfl⟩
      /-
        case intro
        𝕜 : Type u_1
        A : Type u_2
        inst✝⁴ : NontriviallyNormedField 𝕜
        inst✝³ : NonUnitalNormedRing A
        inst✝² : NormedSpace 𝕜 A
        inst✝¹ : SMulCommClass 𝕜 A A
        inst✝ : IsScalarTower 𝕜 A A
        a : DoubleCentralizer 𝕜 A
        ⊢ Membership.mem (setOf fun lr => ∀ (x y : A), Eq (HMul.hMul ((MulOpposite.uno …
      -/
      exact a.central, fun hx => ⟨⟨(x.1, unop x.2), hx⟩, Prod.ext rfl rfl⟩⟩
      /-
        🎉 no goals
      -/


/-- The ring structure is inherited as the pullback under the injective map
`DoubleCentralizer.toProdMulOpposite : 𝓜(𝕜, A) → (A →L[𝕜] A) × (A →L[𝕜] A)ᵐᵒᵖ` -/
instance instRing : Ring 𝓜(𝕜, A) :=
  toProdMulOpposite_injective.ring _ rfl rfl (fun _ _ => rfl) (fun _ _ => rfl) (fun _ => rfl)
    (fun _ _ => rfl) (fun _x _n => Prod.ext rfl <| MulOpposite.op_smul _ _)
    (fun _x _n => Prod.ext rfl <| MulOpposite.op_smul _ _)
    (fun _x _n => Prod.ext rfl <| MulOpposite.op_pow _ _) (fun _ => rfl) fun _ => rfl


/-- The canonical map `DoubleCentralizer.toProd` as an additive group homomorphism. -/
@[simps]
def toProdHom : 𝓜(𝕜, A) →+ (A →L[𝕜] A) × (A →L[𝕜] A) where
  toFun := toProd
  map_zero' := rfl
  map_add' _x _y := rfl


/-- The canonical map `DoubleCentralizer.toProdMulOpposite` as a ring homomorphism. -/
@[simps]
def toProdMulOppositeHom : 𝓜(𝕜, A) →+* (A →L[𝕜] A) × (A →L[𝕜] A)ᵐᵒᵖ where
  toFun := toProdMulOpposite
  map_zero' := rfl
  map_one' := rfl
  map_add' _x _y := rfl
  map_mul' _x _y := rfl


/-- The module structure is inherited as the pullback under the additive group monomorphism
`DoubleCentralizer.toProd : 𝓜(𝕜, A) →+ (A →L[𝕜] A) × (A →L[𝕜] A)` -/
instance instModule {S : Type*} [Semiring S] [Module S A] [SMulCommClass 𝕜 S A]
    [ContinuousConstSMul S A] [IsScalarTower S A A] [SMulCommClass S A A] : Module S 𝓜(𝕜, A) :=
  Function.Injective.module S toProdHom (ext (𝕜 := 𝕜) (A := A)) fun _x _y => rfl

-- TODO: generalize to `Algebra S 𝓜(𝕜, A)` once `ContinuousLinearMap.algebra` is generalized.

instance instAlgebra : Algebra 𝕜 𝓜(𝕜, A) where
  toFun k :=
    { toProd := algebraMap 𝕜 ((A →L[𝕜] A) × (A →L[𝕜] A)) k
      central := fun x y => by
        simp_rw [Prod.algebraMap_apply, Algebra.algebraMap_eq_smul_one, smul_apply, one_apply,
          mul_smul_comm, smul_mul_assoc] }
  map_one' := ext (𝕜 := 𝕜) (A := A) _ _ <| map_one <| algebraMap 𝕜 ((A →L[𝕜] A) × (A →L[𝕜] A))
  map_mul' _ _ :=
    ext (𝕜 := 𝕜) (A := A) _ _ <|
      Prod.ext (map_mul (algebraMap 𝕜 (A →L[𝕜] A)) _ _)
        ((map_mul (algebraMap 𝕜 (A →L[𝕜] A)) _ _).trans (Algebra.commutes _ _))
  map_zero' := ext (𝕜 := 𝕜) (A := A) _ _ <| map_zero <| algebraMap 𝕜 ((A →L[𝕜] A) × (A →L[𝕜] A))
  map_add' _ _ := ext (𝕜 := 𝕜) (A := A) _ _ <|
    map_add (algebraMap 𝕜 ((A →L[𝕜] A) × (A →L[𝕜] A))) _ _
  commutes' _ _ := ext (𝕜 := 𝕜) (A := A) _ _ <|
    Prod.ext (Algebra.commutes _ _) (Algebra.commutes _ _).symm
  smul_def' _ _ := ext (𝕜 := 𝕜) (A := A) _ _ <|
    Prod.ext (Algebra.smul_def _ _) ((Algebra.smul_def _ _).trans <| Algebra.commutes _ _)


@[simp]
theorem algebraMap_toProd (k : 𝕜) : (algebraMap 𝕜 𝓜(𝕜, A) k).toProd = algebraMap 𝕜 _ k :=
  rfl


theorem algebraMap_fst (k : 𝕜) : (algebraMap 𝕜 𝓜(𝕜, A) k).fst = algebraMap 𝕜 _ k :=
  rfl


theorem algebraMap_snd (k : 𝕜) : (algebraMap 𝕜 𝓜(𝕜, A) k).snd = algebraMap 𝕜 _ k :=
  rfl


/-- The star operation on `a : 𝓜(𝕜, A)` is given by
`(star a).toProd = (star ∘ a.snd ∘ star, star ∘ a.fst ∘ star)`. -/
instance instStar : Star 𝓜(𝕜, A) where
  star a :=
    { fst :=
        (((starₗᵢ 𝕜 : A ≃ₗᵢ⋆[𝕜] A) : A →L⋆[𝕜] A).comp a.snd).comp
          ((starₗᵢ 𝕜 : A ≃ₗᵢ⋆[𝕜] A) : A →L⋆[𝕜] A)
      snd :=
        (((starₗᵢ 𝕜 : A ≃ₗᵢ⋆[𝕜] A) : A →L⋆[𝕜] A).comp a.fst).comp
          ((starₗᵢ 𝕜 : A ≃ₗᵢ⋆[𝕜] A) : A →L⋆[𝕜] A)
      central := fun x y => by
        /-
          𝕜 : Type u_1
          A : Type u_2
          inst✝⁸ : NontriviallyNormedField 𝕜
          inst✝⁷ : NonUnitalNormedRing A
          inst✝⁶ : NormedSpace 𝕜 A
          inst✝⁵ : SMulCommClass 𝕜 A A
          inst✝⁴ : IsScalarTower 𝕜 A A
          inst✝³ : StarRing 𝕜
          inst✝² : StarRing A
          inst✝¹ : StarModule 𝕜 A
          inst✝ : NormedStarGroup A
          a : DoubleCentralizer 𝕜 A
          x y : A
          ⊢ Eq (HMul.hMul ({ fst := ((↑{ toLinearEquiv := (starₗᵢ 𝕜).toLinearEquiv, cont …
        -/
        simpa only [star_mul, star_star] using (congr_arg star (a.central (star y) (star x))).symm }
        /-
          🎉 no goals
        -/


@[simp]
theorem star_fst (a : 𝓜(𝕜, A)) (b : A) : (star a).fst b = star (a.snd (star b)) :=
  rfl


@[simp]
theorem star_snd (a : 𝓜(𝕜, A)) (b : A) : (star a).snd b = star (a.fst (star b)) :=
  rfl


instance instStarAddMonoid : StarAddMonoid 𝓜(𝕜, A) :=
  { DoubleCentralizer.instStar with
                                   /-
                                     𝕜 : Type u_1
                                     A : Type u_2
                                     inst✝⁸ : NontriviallyNormedField 𝕜
                                     inst✝⁷ : NonUnitalNormedRing A
                                     inst✝⁶ : NormedSpace 𝕜 A
                                     inst✝⁵ : SMulCommClass 𝕜 A A
                                     inst✝⁴ : IsScalarTower 𝕜 A A
                                     inst✝³ : StarRing 𝕜
                                     inst✝² : StarRing A
                                     inst✝¹ : StarModule 𝕜 A
                                     inst✝ : NormedStarGroup A
                                     x : DoubleCentralizer 𝕜 A
                                     ⊢ Eq (Star.star (Star.star x)) x
                                   -/
                                           /-
                                             🎉 no goals
                                           -/
    star_involutive := fun x => by ext <;> simp only [star_fst, star_snd, star_star]
                                           /-
                                             🎉 no goals
                                           -/
    star_add := fun x y => by
      /-
        𝕜 : Type u_1
        A : Type u_2
        inst✝⁸ : NontriviallyNormedField 𝕜
        inst✝⁷ : NonUnitalNormedRing A
        inst✝⁶ : NormedSpace 𝕜 A
        inst✝⁵ : SMulCommClass 𝕜 A A
        inst✝⁴ : IsScalarTower 𝕜 A A
        inst✝³ : StarRing 𝕜
        inst✝² : StarRing A
        inst✝¹ : StarModule 𝕜 A
        inst✝ : NormedStarGroup A
        x y : DoubleCentralizer 𝕜 A
        ⊢ Eq (Star.star (HAdd.hAdd x y)) (HAdd.hAdd (Star.star x) (Star.star y))
      -/
      ext <;>
        /-
          case h.fst.h
          𝕜 : Type u_1
          A : Type u_2
          inst✝⁸ : NontriviallyNormedField 𝕜
          inst✝⁷ : NonUnitalNormedRing A
          inst✝⁶ : NormedSpace 𝕜 A
          inst✝⁵ : SMulCommClass 𝕜 A A
          inst✝⁴ : IsScalarTower 𝕜 A A
          inst✝³ : StarRing 𝕜
          inst✝² : StarRing A
          inst✝¹ : StarModule 𝕜 A
          inst✝ : NormedStarGroup A
          x y : DoubleCentralizer 𝕜 A
          x✝ : A
          ⊢ Eq ((Star.star (HAdd.hAdd x y)).toProd.1 x✝) ((HAdd.hAdd (Star.star x) (Star …
        -/
        /-
          🎉 no goals
        -/
        simp only [star_fst, star_snd, add_fst, add_snd, ContinuousLinearMap.add_apply, star_add] }
        /-
          🎉 no goals
        -/


instance instStarRing : StarRing 𝓜(𝕜, A) :=
  { DoubleCentralizer.instStarAddMonoid with
    star_mul := fun a b => by
      /-
        𝕜 : Type u_1
        A : Type u_2
        inst✝⁸ : NontriviallyNormedField 𝕜
        inst✝⁷ : NonUnitalNormedRing A
        inst✝⁶ : NormedSpace 𝕜 A
        inst✝⁵ : SMulCommClass 𝕜 A A
        inst✝⁴ : IsScalarTower 𝕜 A A
        inst✝³ : StarRing 𝕜
        inst✝² : StarRing A
        inst✝¹ : StarModule 𝕜 A
        inst✝ : NormedStarGroup A
        a b : DoubleCentralizer 𝕜 A
        ⊢ Eq (Star.star (HMul.hMul a b)) (HMul.hMul (Star.star b) (Star.star a))
      -/
      ext <;>
        simp only [star_fst, star_snd, mul_fst, mul_snd, star_star, ContinuousLinearMap.coe_mul,
          Function.comp_apply] }


instance instStarModule : StarModule 𝕜 𝓜(𝕜, A) :=
  { DoubleCentralizer.instStarAddMonoid (𝕜 := 𝕜) (A := A) with
                               /-
                                 𝕜 : Type u_1
                                 A : Type u_2
                                 inst✝⁸ : NontriviallyNormedField 𝕜
                                 inst✝⁷ : NonUnitalNormedRing A
                                 inst✝⁶ : NormedSpace 𝕜 A
                                 inst✝⁵ : SMulCommClass 𝕜 A A
                                 inst✝⁴ : IsScalarTower 𝕜 A A
                                 inst✝³ : StarRing 𝕜
                                 inst✝² : StarRing A
                                 inst✝¹ : StarModule 𝕜 A
                                 inst✝ : NormedStarGroup A
                                 k : 𝕜
                                 a : DoubleCentralizer 𝕜 A
                                 ⊢ Eq (Star.star (HSMul.hSMul k a)) (HSMul.hSMul (Star.star k) (Star.star a))
                               -/
                                       /-
                                         🎉 no goals
                                       -/
    star_smul := fun k a => by ext <;> exact star_smul _ _ }
                                       /-
                                         🎉 no goals
                                       -/


/-- The natural coercion of `A` into `𝓜(𝕜, A)` given by sending `a : A` to the pair of linear
maps `Lₐ Rₐ : A →L[𝕜] A` given by left- and right-multiplication by `a`, respectively.

Warning: if `A = 𝕜`, then this is a coercion which is not definitionally equal to the
`algebraMap 𝕜 𝓜(𝕜, 𝕜)` coercion, but these are propositionally equal. See
`DoubleCentralizer.coe_eq_algebraMap` below. -/
-- Porting note: added `noncomputable`; IR check does not recognise `ContinuousLinearMap.mul`
@[coe]
protected noncomputable def coe (a : A) : 𝓜(𝕜, A) :=
  { fst := ContinuousLinearMap.mul 𝕜 A a
    snd := (ContinuousLinearMap.mul 𝕜 A).flip a
    central := fun _x _y => mul_assoc _ _ _ }


/-- The natural coercion of `A` into `𝓜(𝕜, A)` given by sending `a : A` to the pair of linear
maps `Lₐ Rₐ : A →L[𝕜] A` given by left- and right-multiplication by `a`, respectively.

Warning: if `A = 𝕜`, then this is a coercion which is not definitionally equal to the
`algebraMap 𝕜 𝓜(𝕜, 𝕜)` coercion, but these are propositionally equal. See
`DoubleCentralizer.coe_eq_algebraMap` below. -/
noncomputable instance : CoeTC A 𝓜(𝕜, A) where
  coe := DoubleCentralizer.coe 𝕜


@[simp, norm_cast]
theorem coe_fst (a : A) : (a : 𝓜(𝕜, A)).fst = ContinuousLinearMap.mul 𝕜 A a :=
  rfl


@[simp, norm_cast]
theorem coe_snd (a : A) : (a : 𝓜(𝕜, A)).snd = (ContinuousLinearMap.mul 𝕜 A).flip a :=
  rfl


theorem coe_eq_algebraMap : (DoubleCentralizer.coe 𝕜 : 𝕜 → 𝓜(𝕜, 𝕜)) = algebraMap 𝕜 𝓜(𝕜, 𝕜) := by
  /-
    𝕜 : Type u_1
    inst✝ : NontriviallyNormedField 𝕜
    ⊢ Eq ↑𝕜 ⇑(algebraMap 𝕜 (DoubleCentralizer 𝕜 𝕜))
  -/
  ext x : 3
    /-
      case h.h.fst
      𝕜 : Type u_1
      inst✝ : NontriviallyNormedField 𝕜
      x : 𝕜
      ⊢ Eq (↑𝕜 x).toProd.1 ((algebraMap 𝕜 (DoubleCentralizer 𝕜 𝕜)) x).toProd.1
    -/
  · rfl -- `fst` is defeq
    /-
      🎉 no goals
    -/
    /-
      case h.h.snd
      𝕜 : Type u_1
      inst✝ : NontriviallyNormedField 𝕜
      x : 𝕜
      ⊢ Eq (↑𝕜 x).toProd.2 ((algebraMap 𝕜 (DoubleCentralizer 𝕜 𝕜)) x).toProd.2
    -/
  · refine ContinuousLinearMap.ext fun y => ?_
    /-
      case h.h.snd
      𝕜 : Type u_1
      inst✝ : NontriviallyNormedField 𝕜
      x y : 𝕜
      ⊢ Eq ((↑𝕜 x).toProd.2 y) (((algebraMap 𝕜 (DoubleCentralizer 𝕜 𝕜)) x).toProd.2 y)
    -/
    exact mul_comm y x  -- `snd` multiplies on the wrong side
    /-
      🎉 no goals
    -/


/-- The coercion of an algebra into its multiplier algebra as a non-unital star algebra
homomorphism. -/
@[simps]
noncomputable def coeHom [StarRing 𝕜] [StarRing A] [StarModule 𝕜 A] [NormedStarGroup A] :
    A →⋆ₙₐ[𝕜] 𝓜(𝕜, A) where
  toFun a := a
  map_smul' _ _ := ext _ _ _ _ <| Prod.ext (map_smul _ _ _) (map_smul _ _ _)
  map_zero' := ext _ _ _ _ <| Prod.ext (map_zero _) (map_zero _)
  map_add' _ _ := ext _ _ _ _ <| Prod.ext (map_add _ _ _) (map_add _ _ _)
  map_mul' _ _ := ext _ _ _ _ <| Prod.ext
    (ContinuousLinearMap.ext fun _ => (mul_assoc _ _ _))
    (ContinuousLinearMap.ext fun _ => (mul_assoc _ _ _).symm)
  map_star' _ := ext _ _ _ _ <| Prod.ext
    (ContinuousLinearMap.ext fun _ => (star_star_mul _ _).symm)
    (ContinuousLinearMap.ext fun _ => (star_mul_star _ _).symm)


/-- The normed group structure is inherited as the pullback under the ring monomorphism
`DoubleCentralizer.toProdMulOppositeHom : 𝓜(𝕜, A) →+* (A →L[𝕜] A) × (A →L[𝕜] A)ᵐᵒᵖ`. -/
noncomputable instance : NormedRing 𝓜(𝕜, A) :=
  NormedRing.induced _ _ (toProdMulOppositeHom : 𝓜(𝕜, A) →+* (A →L[𝕜] A) × (A →L[𝕜] A)ᵐᵒᵖ)
        /-
          𝕜 : Type u_1
          A : Type u_2
          inst✝⁴ : NontriviallyNormedField 𝕜
          inst✝³ : NonUnitalNormedRing A
          inst✝² : NormedSpace 𝕜 A
          inst✝¹ : SMulCommClass 𝕜 A A
          inst✝ : IsScalarTower 𝕜 A A
          ⊢ Function.Injective ⇑DoubleCentralizer.toProdMulOppositeHom
        -/
    (by simpa using toProdMulOpposite_injective)
        /-
          🎉 no goals
        -/

-- even though the definition is actually in terms of `DoubleCentralizer.toProdMulOpposite`, we
-- choose to see through that here to avoid `MulOpposite.op` appearing.

theorem norm_def (a : 𝓜(𝕜, A)) : ‖a‖ = ‖toProdHom a‖ :=
  rfl


theorem nnnorm_def (a : 𝓜(𝕜, A)) : ‖a‖₊ = ‖toProdHom a‖₊ :=
  rfl


theorem norm_def' (a : 𝓜(𝕜, A)) : ‖a‖ = ‖toProdMulOppositeHom a‖ :=
  rfl


theorem nnnorm_def' (a : 𝓜(𝕜, A)) : ‖a‖₊ = ‖toProdMulOppositeHom a‖₊ :=
  rfl


instance instNormedSpace : NormedSpace 𝕜 𝓜(𝕜, A) :=
  { DoubleCentralizer.instModule with
    norm_smul_le := fun k a => (norm_smul_le k a.toProdMulOpposite : _) }


instance instNormedAlgebra : NormedAlgebra 𝕜 𝓜(𝕜, A) :=
  { DoubleCentralizer.instAlgebra, DoubleCentralizer.instNormedSpace with }


theorem isUniformEmbedding_toProdMulOpposite :
    IsUniformEmbedding (toProdMulOpposite (𝕜 := 𝕜) (A := A)) :=
  isUniformEmbedding_comap toProdMulOpposite_injective


@[deprecated (since := "2024-10-01")]
alias uniformEmbedding_toProdMulOpposite := isUniformEmbedding_toProdMulOpposite


instance [CompleteSpace A] : CompleteSpace 𝓜(𝕜, A) := by
  /-
    𝕜 : Type u_1
    A : Type u_2
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : NonUnitalNormedRing A
    inst✝³ : NormedSpace 𝕜 A
    inst✝² : SMulCommClass 𝕜 A A
    inst✝¹ : IsScalarTower 𝕜 A A
    inst✝ : CompleteSpace A
    ⊢ CompleteSpace (DoubleCentralizer 𝕜 A)
  -/
  rw [completeSpace_iff_isComplete_range isUniformEmbedding_toProdMulOpposite.isUniformInducing]
  /-
    𝕜 : Type u_1
    A : Type u_2
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : NonUnitalNormedRing A
    inst✝³ : NormedSpace 𝕜 A
    inst✝² : SMulCommClass 𝕜 A A
    inst✝¹ : IsScalarTower 𝕜 A A
    inst✝ : CompleteSpace A
    ⊢ IsComplete (Set.range DoubleCentralizer.toProdMulOpposite)
  -/
  apply IsClosed.isComplete
  /-
    case h
    𝕜 : Type u_1
    A : Type u_2
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : NonUnitalNormedRing A
    inst✝³ : NormedSpace 𝕜 A
    inst✝² : SMulCommClass 𝕜 A A
    inst✝¹ : IsScalarTower 𝕜 A A
    inst✝ : CompleteSpace A
    ⊢ IsClosed (Set.range DoubleCentralizer.toProdMulOpposite)
  -/
  simp only [range_toProdMulOpposite, Set.setOf_forall]
  /-
    case h
    𝕜 : Type u_1
    A : Type u_2
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : NonUnitalNormedRing A
    inst✝³ : NormedSpace 𝕜 A
    inst✝² : SMulCommClass 𝕜 A A
    inst✝¹ : IsScalarTower 𝕜 A A
    inst✝ : CompleteSpace A
    ⊢ IsClosed (Set.iInter fun i => Set.iInter fun i_1 => setOf fun x => Eq (HMul. …
  -/
  refine isClosed_iInter fun x => isClosed_iInter fun y => isClosed_eq ?_ ?_
  · exact
      ((ContinuousLinearMap.apply 𝕜 A _).continuous.comp <| continuous_unop.comp continuous_snd).mul
        continuous_const
  /-
    case h.refine_2
    𝕜 : Type u_1
    A : Type u_2
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : NonUnitalNormedRing A
    inst✝³ : NormedSpace 𝕜 A
    inst✝² : SMulCommClass 𝕜 A A
    inst✝¹ : IsScalarTower 𝕜 A A
    inst✝ : CompleteSpace A
    x y : A
    ⊢ Continuous fun x_1 => HMul.hMul x (x_1.1 y)
  -/
  exact continuous_const.mul ((ContinuousLinearMap.apply 𝕜 A _).continuous.comp continuous_fst)
  /-
    🎉 no goals
  -/


/-- For `a : 𝓜(𝕜, A)`, the norms of `a.fst` and `a.snd` coincide, and hence these
also coincide with `‖a‖` which is `max (‖a.fst‖) (‖a.snd‖)`. -/
theorem norm_fst_eq_snd (a : 𝓜(𝕜, A)) : ‖a.fst‖ = ‖a.snd‖ := by
  -- a handy lemma for this proof
  have h0 : ∀ f : A →L[𝕜] A, ∀ C : ℝ≥0, (∀ b : A, ‖f b‖₊ ^ 2 ≤ C * ‖f b‖₊ * ‖b‖₊) → ‖f‖₊ ≤ C := by
    intro f C h
    have h1 : ∀ b, C * ‖f b‖₊ * ‖b‖₊ ≤ C * ‖f‖₊ * ‖b‖₊ ^ 2 := by
      intro b
      convert mul_le_mul_right' (mul_le_mul_left' (f.le_opNNNorm b) C) ‖b‖₊ using 1
      ring
    have := NNReal.div_le_of_le_mul <| f.opNNNorm_le_bound _ <| by
      simpa only [sqrt_sq, sqrt_mul] using fun b ↦ sqrt_le_sqrt.2 <| (h b).trans (h1 b)
    convert NNReal.rpow_le_rpow this two_pos.le
    · simp only [NNReal.rpow_two, div_pow, sq_sqrt]
      simp only [sq, mul_self_div_self]
    · simp only [NNReal.rpow_two, sq_sqrt]
  have h1 : ∀ b, ‖a.fst b‖₊ ^ 2 ≤ ‖a.snd‖₊ * ‖a.fst b‖₊ * ‖b‖₊ := by
    intro b
    calc
      ‖a.fst b‖₊ ^ 2 = ‖star (a.fst b) * a.fst b‖₊ := by
        simpa only [← sq] using CStarRing.nnnorm_star_mul_self.symm
      _ ≤ ‖a.snd (star (a.fst b))‖₊ * ‖b‖₊ := (a.central (star (a.fst b)) b ▸ nnnorm_mul_le _ _)
      _ ≤ ‖a.snd‖₊ * ‖a.fst b‖₊ * ‖b‖₊ :=
        nnnorm_star (a.fst b) ▸ mul_le_mul_right' (a.snd.le_opNNNorm _) _

  have h2 : ∀ b, ‖a.snd b‖₊ ^ 2 ≤ ‖a.fst‖₊ * ‖a.snd b‖₊ * ‖b‖₊ := by
    intro b
    calc
      ‖a.snd b‖₊ ^ 2 = ‖a.snd b * star (a.snd b)‖₊ := by
        simpa only [← sq] using CStarRing.nnnorm_self_mul_star.symm
      _ ≤ ‖b‖₊ * ‖a.fst (star (a.snd b))‖₊ :=
        ((a.central b (star (a.snd b))).symm ▸ nnnorm_mul_le _ _)
      _ = ‖a.fst (star (a.snd b))‖₊ * ‖b‖₊ := mul_comm _ _
      _ ≤ ‖a.fst‖₊ * ‖a.snd b‖₊ * ‖b‖₊ :=
        nnnorm_star (a.snd b) ▸ mul_le_mul_right' (a.fst.le_opNNNorm _) _

  /-
    𝕜 : Type u_1
    A : Type u_2
    inst✝⁶ : NontriviallyNormedField 𝕜
    inst✝⁵ : NonUnitalNormedRing A
    inst✝⁴ : NormedSpace 𝕜 A
    inst✝³ : SMulCommClass 𝕜 A A
    inst✝² : IsScalarTower 𝕜 A A
    inst✝¹ : StarRing A
    inst✝ : CStarRing A
    a : DoubleCentralizer 𝕜 A
    h0 : ∀ (f : ContinuousLinearMap (RingHom.id 𝕜) A A) (C : NNReal), (∀ (b : A),  …
    h1 : ∀ (b : A), LE.le (HPow.hPow (NNNorm.nnnorm (a.toProd.1 b)) 2) (HMul.hMul  …
    h2 : ∀ (b : A), LE.le (HPow.hPow (NNNorm.nnnorm (a.toProd.2 b)) 2) (HMul.hMul  …
    ⊢ Eq (Norm.norm a.toProd.1) (Norm.norm a.toProd.2)
  -/
  exact le_antisymm (h0 _ _ h1) (h0 _ _ h2)
  /-
    🎉 no goals
  -/


theorem nnnorm_fst_eq_snd (a : 𝓜(𝕜, A)) : ‖a.fst‖₊ = ‖a.snd‖₊ :=
  Subtype.ext <| norm_fst_eq_snd a


@[simp]
theorem norm_fst (a : 𝓜(𝕜, A)) : ‖a.fst‖ = ‖a‖ := by
  /-
    𝕜 : Type u_1
    A : Type u_2
    inst✝⁶ : NontriviallyNormedField 𝕜
    inst✝⁵ : NonUnitalNormedRing A
    inst✝⁴ : NormedSpace 𝕜 A
    inst✝³ : SMulCommClass 𝕜 A A
    inst✝² : IsScalarTower 𝕜 A A
    inst✝¹ : StarRing A
    inst✝ : CStarRing A
    a : DoubleCentralizer 𝕜 A
    ⊢ Eq (Norm.norm a.toProd.1) (Norm.norm a)
  -/
  simp only [norm_def, toProdHom_apply, Prod.norm_def, norm_fst_eq_snd, max_eq_right le_rfl]
  /-
    🎉 no goals
  -/


@[simp]
                                                     /-
                                                       𝕜 : Type u_1
                                                       A : Type u_2
                                                       inst✝⁶ : NontriviallyNormedField 𝕜
                                                       inst✝⁵ : NonUnitalNormedRing A
                                                       inst✝⁴ : NormedSpace 𝕜 A
                                                       inst✝³ : SMulCommClass 𝕜 A A
                                                       inst✝² : IsScalarTower 𝕜 A A
                                                       inst✝¹ : StarRing A
                                                       inst✝ : CStarRing A
                                                       a : DoubleCentralizer 𝕜 A
                                                       ⊢ Eq (Norm.norm a.toProd.2) (Norm.norm a)
                                                     -/
theorem norm_snd (a : 𝓜(𝕜, A)) : ‖a.snd‖ = ‖a‖ := by rw [← norm_fst, norm_fst_eq_snd]
                                                     /-
                                                       🎉 no goals
                                                     -/


@[simp]
theorem nnnorm_fst (a : 𝓜(𝕜, A)) : ‖a.fst‖₊ = ‖a‖₊ :=
  Subtype.ext (norm_fst a)


@[simp]
theorem nnnorm_snd (a : 𝓜(𝕜, A)) : ‖a.snd‖₊ = ‖a‖₊ :=
  Subtype.ext (norm_snd a)


instance instCStarRing : CStarRing 𝓜(𝕜, A) where
  norm_mul_self_le := fun (a : 𝓜(𝕜, A)) => le_of_eq <| Eq.symm <| congr_arg ((↑) : ℝ≥0 → ℝ) <|
    show ‖star a * a‖₊ = ‖a‖₊ * ‖a‖₊ by
    /- The essence of the argument is this: let `a = (L,R)` and recall `‖a‖ = ‖L‖`.
    `star a = (star ∘ R ∘ star, star ∘ L ∘ star)`. Then for any `x y : A`, we have
    `‖star a * a‖ = ‖(star a * a).snd‖ = ‖R (star (L (star x))) * y‖ = ‖star (L (star x)) * L y‖`
    Now, on the one hand,
    `‖star (L (star x)) * L y‖ ≤ ‖star (L (star x))‖ * ‖L y‖ = ‖L (star x)‖ * ‖L y‖ ≤ ‖L‖ ^ 2`
    whenever `‖x‖, ‖y‖ ≤ 1`, so the supremum over all such `x, y` is at most `‖L‖ ^ 2`.
    On the other hand, for any `‖z‖ ≤ 1`, we may choose `x := star z` and `y := z` to get:
    `‖star (L (star x)) * L y‖ = ‖star (L z) * (L z)‖ = ‖L z‖ ^ 2`, and taking the supremum over
    all such `z` yields that the supremum is at least `‖L‖ ^ 2`. It is the latter part of the
    argument where `DenselyNormedField 𝕜` is required (for `sSup_unitClosedBall_eq_nnnorm`). -/
      have hball : (Metric.closedBall (0 : A) 1).Nonempty :=
        Metric.nonempty_closedBall.2 zero_le_one
      have key :
        ∀ x y, ‖x‖₊ ≤ 1 → ‖y‖₊ ≤ 1 → ‖a.snd (star (a.fst (star x))) * y‖₊ ≤ ‖a‖₊ * ‖a‖₊ := by
        intro x y hx hy
        rw [a.central]
        calc
          ‖star (a.fst (star x)) * a.fst y‖₊ ≤ ‖a.fst (star x)‖₊ * ‖a.fst y‖₊ :=
            nnnorm_star (a.fst (star x)) ▸ nnnorm_mul_le _ _
          _ ≤ ‖a.fst‖₊ * 1 * (‖a.fst‖₊ * 1) :=
            (mul_le_mul' (a.fst.le_opNorm_of_le ((nnnorm_star x).trans_le hx))
              (a.fst.le_opNorm_of_le hy))
          _ ≤ ‖a‖₊ * ‖a‖₊ := by simp only [mul_one, nnnorm_fst, le_rfl]
      /-
        𝕜 : Type u_1
        A : Type u_2
        inst✝⁸ : DenselyNormedField 𝕜
        inst✝⁷ : StarRing 𝕜
        inst✝⁶ : NonUnitalNormedRing A
        inst✝⁵ : StarRing A
        inst✝⁴ : CStarRing A
        inst✝³ : NormedSpace 𝕜 A
        inst✝² : SMulCommClass 𝕜 A A
        inst✝¹ : IsScalarTower 𝕜 A A
        inst✝ : StarModule 𝕜 A
        a : DoubleCentralizer 𝕜 A
        hball : (Metric.closedBall 0 1).Nonempty
        key : ∀ (x y : A), LE.le (NNNorm.nnnorm x) 1 → LE.le (NNNorm.nnnorm y) 1 → LE. …
        ⊢ Eq (NNNorm.nnnorm (HMul.hMul (Star.star a) a)) (HMul.hMul (NNNorm.nnnorm a)  …
      -/
      rw [← nnnorm_snd]
      /-
        𝕜 : Type u_1
        A : Type u_2
        inst✝⁸ : DenselyNormedField 𝕜
        inst✝⁷ : StarRing 𝕜
        inst✝⁶ : NonUnitalNormedRing A
        inst✝⁵ : StarRing A
        inst✝⁴ : CStarRing A
        inst✝³ : NormedSpace 𝕜 A
        inst✝² : SMulCommClass 𝕜 A A
        inst✝¹ : IsScalarTower 𝕜 A A
        inst✝ : StarModule 𝕜 A
        a : DoubleCentralizer 𝕜 A
        hball : (Metric.closedBall 0 1).Nonempty
        key : ∀ (x y : A), LE.le (NNNorm.nnnorm x) 1 → LE.le (NNNorm.nnnorm y) 1 → LE. …
        ⊢ Eq (NNNorm.nnnorm (HMul.hMul (Star.star a) a).toProd.2) (HMul.hMul (NNNorm.n …
      -/
      simp only [mul_snd, ← sSup_unitClosedBall_eq_nnnorm, star_snd, mul_apply]
      /-
        𝕜 : Type u_1
        A : Type u_2
        inst✝⁸ : DenselyNormedField 𝕜
        inst✝⁷ : StarRing 𝕜
        inst✝⁶ : NonUnitalNormedRing A
        inst✝⁵ : StarRing A
        inst✝⁴ : CStarRing A
        inst✝³ : NormedSpace 𝕜 A
        inst✝² : SMulCommClass 𝕜 A A
        inst✝¹ : IsScalarTower 𝕜 A A
        inst✝ : StarModule 𝕜 A
        a : DoubleCentralizer 𝕜 A
        hball : (Metric.closedBall 0 1).Nonempty
        key : ∀ (x y : A), LE.le (NNNorm.nnnorm x) 1 → LE.le (NNNorm.nnnorm y) 1 → LE. …
        ⊢ Eq (SupSet.sSup (Set.image (fun a_1 => NNNorm.nnnorm (a.toProd.2 (Star.star  …
      -/
      simp only [← @opNNNorm_mul_apply 𝕜 _ A]
      /-
        𝕜 : Type u_1
        A : Type u_2
        inst✝⁸ : DenselyNormedField 𝕜
        inst✝⁷ : StarRing 𝕜
        inst✝⁶ : NonUnitalNormedRing A
        inst✝⁵ : StarRing A
        inst✝⁴ : CStarRing A
        inst✝³ : NormedSpace 𝕜 A
        inst✝² : SMulCommClass 𝕜 A A
        inst✝¹ : IsScalarTower 𝕜 A A
        inst✝ : StarModule 𝕜 A
        a : DoubleCentralizer 𝕜 A
        hball : (Metric.closedBall 0 1).Nonempty
        key : ∀ (x y : A), LE.le (NNNorm.nnnorm x) 1 → LE.le (NNNorm.nnnorm y) 1 → LE. …
        ⊢ Eq (SupSet.sSup (Set.image (fun a_1 => NNNorm.nnnorm ((ContinuousLinearMap.m …
      -/
      simp only [← sSup_unitClosedBall_eq_nnnorm, mul_apply']
      /-
        𝕜 : Type u_1
        A : Type u_2
        inst✝⁸ : DenselyNormedField 𝕜
        inst✝⁷ : StarRing 𝕜
        inst✝⁶ : NonUnitalNormedRing A
        inst✝⁵ : StarRing A
        inst✝⁴ : CStarRing A
        inst✝³ : NormedSpace 𝕜 A
        inst✝² : SMulCommClass 𝕜 A A
        inst✝¹ : IsScalarTower 𝕜 A A
        inst✝ : StarModule 𝕜 A
        a : DoubleCentralizer 𝕜 A
        hball : (Metric.closedBall 0 1).Nonempty
        key : ∀ (x y : A), LE.le (NNNorm.nnnorm x) 1 → LE.le (NNNorm.nnnorm y) 1 → LE. …
        ⊢ Eq (SupSet.sSup (Set.image (fun a_1 => SupSet.sSup (Set.image (fun a_2 => NN …
      -/
      refine csSup_eq_of_forall_le_of_forall_lt_exists_gt (hball.image _) ?_ fun r hr => ?_
        /-
          case refine_1
          𝕜 : Type u_1
          A : Type u_2
          inst✝⁸ : DenselyNormedField 𝕜
          inst✝⁷ : StarRing 𝕜
          inst✝⁶ : NonUnitalNormedRing A
          inst✝⁵ : StarRing A
          inst✝⁴ : CStarRing A
          inst✝³ : NormedSpace 𝕜 A
          inst✝² : SMulCommClass 𝕜 A A
          inst✝¹ : IsScalarTower 𝕜 A A
          inst✝ : StarModule 𝕜 A
          a : DoubleCentralizer 𝕜 A
          hball : (Metric.closedBall 0 1).Nonempty
          key : ∀ (x y : A), LE.le (NNNorm.nnnorm x) 1 → LE.le (NNNorm.nnnorm y) 1 → LE. …
          ⊢ ∀ (a_1 : NNReal), Membership.mem (Set.image (fun a_2 => SupSet.sSup (Set.ima …
        -/
      · rintro - ⟨x, hx, rfl⟩
        /-
          case refine_1.intro.intro
          𝕜 : Type u_1
          A : Type u_2
          inst✝⁸ : DenselyNormedField 𝕜
          inst✝⁷ : StarRing 𝕜
          inst✝⁶ : NonUnitalNormedRing A
          inst✝⁵ : StarRing A
          inst✝⁴ : CStarRing A
          inst✝³ : NormedSpace 𝕜 A
          inst✝² : SMulCommClass 𝕜 A A
          inst✝¹ : IsScalarTower 𝕜 A A
          inst✝ : StarModule 𝕜 A
          a : DoubleCentralizer 𝕜 A
          hball : (Metric.closedBall 0 1).Nonempty
          key : ∀ (x y : A), LE.le (NNNorm.nnnorm x) 1 → LE.le (NNNorm.nnnorm y) 1 → LE. …
          x : A
          hx : Membership.mem (Metric.closedBall 0 1) x
          ⊢ LE.le ((fun a_1 => SupSet.sSup (Set.image (fun a_2 => NNNorm.nnnorm (HMul.hM …
        -/
        refine csSup_le (hball.image _) ?_
        /-
          case refine_1.intro.intro
          𝕜 : Type u_1
          A : Type u_2
          inst✝⁸ : DenselyNormedField 𝕜
          inst✝⁷ : StarRing 𝕜
          inst✝⁶ : NonUnitalNormedRing A
          inst✝⁵ : StarRing A
          inst✝⁴ : CStarRing A
          inst✝³ : NormedSpace 𝕜 A
          inst✝² : SMulCommClass 𝕜 A A
          inst✝¹ : IsScalarTower 𝕜 A A
          inst✝ : StarModule 𝕜 A
          a : DoubleCentralizer 𝕜 A
          hball : (Metric.closedBall 0 1).Nonempty
          key : ∀ (x y : A), LE.le (NNNorm.nnnorm x) 1 → LE.le (NNNorm.nnnorm y) 1 → LE. …
          x : A
          hx : Membership.mem (Metric.closedBall 0 1) x
          ⊢ ∀ (b : NNReal), Membership.mem (Set.image (fun a_1 => NNNorm.nnnorm (HMul.hM …
        -/
        rintro - ⟨y, hy, rfl⟩
        /-
          case refine_1.intro.intro.intro.intro
          𝕜 : Type u_1
          A : Type u_2
          inst✝⁸ : DenselyNormedField 𝕜
          inst✝⁷ : StarRing 𝕜
          inst✝⁶ : NonUnitalNormedRing A
          inst✝⁵ : StarRing A
          inst✝⁴ : CStarRing A
          inst✝³ : NormedSpace 𝕜 A
          inst✝² : SMulCommClass 𝕜 A A
          inst✝¹ : IsScalarTower 𝕜 A A
          inst✝ : StarModule 𝕜 A
          a : DoubleCentralizer 𝕜 A
          hball : (Metric.closedBall 0 1).Nonempty
          key : ∀ (x y : A), LE.le (NNNorm.nnnorm x) 1 → LE.le (NNNorm.nnnorm y) 1 → LE. …
          x : A
          hx : Membership.mem (Metric.closedBall 0 1) x
          y : A
          hy : Membership.mem (Metric.closedBall 0 1) y
          ⊢ LE.le ((fun a_1 => NNNorm.nnnorm (HMul.hMul (a.toProd.2 (Star.star (a.toProd …
        -/
        exact key x y (mem_closedBall_zero_iff.1 hx) (mem_closedBall_zero_iff.1 hy)
        /-
          🎉 no goals
        -/
        /-
          case refine_2
          𝕜 : Type u_1
          A : Type u_2
          inst✝⁸ : DenselyNormedField 𝕜
          inst✝⁷ : StarRing 𝕜
          inst✝⁶ : NonUnitalNormedRing A
          inst✝⁵ : StarRing A
          inst✝⁴ : CStarRing A
          inst✝³ : NormedSpace 𝕜 A
          inst✝² : SMulCommClass 𝕜 A A
          inst✝¹ : IsScalarTower 𝕜 A A
          inst✝ : StarModule 𝕜 A
          a : DoubleCentralizer 𝕜 A
          hball : (Metric.closedBall 0 1).Nonempty
          key : ∀ (x y : A), LE.le (NNNorm.nnnorm x) 1 → LE.le (NNNorm.nnnorm y) 1 → LE. …
          r : NNReal
          hr : LT.lt r (HMul.hMul (NNNorm.nnnorm a) (NNNorm.nnnorm a))
          ⊢ Exists fun a_1 => And (Membership.mem (Set.image (fun a_2 => SupSet.sSup (Se …
        -/
      · simp only [Set.mem_image, Set.mem_setOf_eq, exists_prop, exists_exists_and_eq_and]
        /-
          case refine_2
          𝕜 : Type u_1
          A : Type u_2
          inst✝⁸ : DenselyNormedField 𝕜
          inst✝⁷ : StarRing 𝕜
          inst✝⁶ : NonUnitalNormedRing A
          inst✝⁵ : StarRing A
          inst✝⁴ : CStarRing A
          inst✝³ : NormedSpace 𝕜 A
          inst✝² : SMulCommClass 𝕜 A A
          inst✝¹ : IsScalarTower 𝕜 A A
          inst✝ : StarModule 𝕜 A
          a : DoubleCentralizer 𝕜 A
          hball : (Metric.closedBall 0 1).Nonempty
          key : ∀ (x y : A), LE.le (NNNorm.nnnorm x) 1 → LE.le (NNNorm.nnnorm y) 1 → LE. …
          r : NNReal
          hr : LT.lt r (HMul.hMul (NNNorm.nnnorm a) (NNNorm.nnnorm a))
          ⊢ Exists fun a_1 => And (Membership.mem (Metric.closedBall 0 1) a_1) (LT.lt r  …
        -/
        have hr' : NNReal.sqrt r < ‖a‖₊ := ‖a‖₊.sqrt_mul_self ▸ NNReal.sqrt_lt_sqrt.2 hr
        /-
          case refine_2
          𝕜 : Type u_1
          A : Type u_2
          inst✝⁸ : DenselyNormedField 𝕜
          inst✝⁷ : StarRing 𝕜
          inst✝⁶ : NonUnitalNormedRing A
          inst✝⁵ : StarRing A
          inst✝⁴ : CStarRing A
          inst✝³ : NormedSpace 𝕜 A
          inst✝² : SMulCommClass 𝕜 A A
          inst✝¹ : IsScalarTower 𝕜 A A
          inst✝ : StarModule 𝕜 A
          a : DoubleCentralizer 𝕜 A
          hball : (Metric.closedBall 0 1).Nonempty
          key : ∀ (x y : A), LE.le (NNNorm.nnnorm x) 1 → LE.le (NNNorm.nnnorm y) 1 → LE. …
          r : NNReal
          hr : LT.lt r (HMul.hMul (NNNorm.nnnorm a) (NNNorm.nnnorm a))
          hr' : LT.lt (NNReal.sqrt r) (NNNorm.nnnorm a)
          ⊢ Exists fun a_1 => And (Membership.mem (Metric.closedBall 0 1) a_1) (LT.lt r  …
        -/
        simp_rw [← nnnorm_fst, ← sSup_unitClosedBall_eq_nnnorm] at hr'
        /-
          case refine_2
          𝕜 : Type u_1
          A : Type u_2
          inst✝⁸ : DenselyNormedField 𝕜
          inst✝⁷ : StarRing 𝕜
          inst✝⁶ : NonUnitalNormedRing A
          inst✝⁵ : StarRing A
          inst✝⁴ : CStarRing A
          inst✝³ : NormedSpace 𝕜 A
          inst✝² : SMulCommClass 𝕜 A A
          inst✝¹ : IsScalarTower 𝕜 A A
          inst✝ : StarModule 𝕜 A
          a : DoubleCentralizer 𝕜 A
          hball : (Metric.closedBall 0 1).Nonempty
          key : ∀ (x y : A), LE.le (NNNorm.nnnorm x) 1 → LE.le (NNNorm.nnnorm y) 1 → LE. …
          r : NNReal
          hr : LT.lt r (HMul.hMul (NNNorm.nnnorm a) (NNNorm.nnnorm a))
          hr' : LT.lt (NNReal.sqrt r) (SupSet.sSup (Set.image (fun x => NNNorm.nnnorm (a …
          ⊢ Exists fun a_1 => And (Membership.mem (Metric.closedBall 0 1) a_1) (LT.lt r  …
        -/
        obtain ⟨_, ⟨x, hx, rfl⟩, hxr⟩ := exists_lt_of_lt_csSup (hball.image _) hr'
        /-
          case refine_2.intro.intro.intro.intro
          𝕜 : Type u_1
          A : Type u_2
          inst✝⁸ : DenselyNormedField 𝕜
          inst✝⁷ : StarRing 𝕜
          inst✝⁶ : NonUnitalNormedRing A
          inst✝⁵ : StarRing A
          inst✝⁴ : CStarRing A
          inst✝³ : NormedSpace 𝕜 A
          inst✝² : SMulCommClass 𝕜 A A
          inst✝¹ : IsScalarTower 𝕜 A A
          inst✝ : StarModule 𝕜 A
          a : DoubleCentralizer 𝕜 A
          hball : (Metric.closedBall 0 1).Nonempty
          key : ∀ (x y : A), LE.le (NNNorm.nnnorm x) 1 → LE.le (NNNorm.nnnorm y) 1 → LE. …
          r : NNReal
          hr : LT.lt r (HMul.hMul (NNNorm.nnnorm a) (NNNorm.nnnorm a))
          hr' : LT.lt (NNReal.sqrt r) (SupSet.sSup (Set.image (fun x => NNNorm.nnnorm (a …
          x : A
          hx : Membership.mem (Metric.closedBall 0 1) x
          hxr : LT.lt (NNReal.sqrt r) ((fun x => NNNorm.nnnorm (a.toProd.1 x)) x)
          ⊢ Exists fun a_1 => And (Membership.mem (Metric.closedBall 0 1) a_1) (LT.lt r  …
        -/
        have hx' : ‖x‖₊ ≤ 1 := mem_closedBall_zero_iff.1 hx
        /-
          case refine_2.intro.intro.intro.intro
          𝕜 : Type u_1
          A : Type u_2
          inst✝⁸ : DenselyNormedField 𝕜
          inst✝⁷ : StarRing 𝕜
          inst✝⁶ : NonUnitalNormedRing A
          inst✝⁵ : StarRing A
          inst✝⁴ : CStarRing A
          inst✝³ : NormedSpace 𝕜 A
          inst✝² : SMulCommClass 𝕜 A A
          inst✝¹ : IsScalarTower 𝕜 A A
          inst✝ : StarModule 𝕜 A
          a : DoubleCentralizer 𝕜 A
          hball : (Metric.closedBall 0 1).Nonempty
          key : ∀ (x y : A), LE.le (NNNorm.nnnorm x) 1 → LE.le (NNNorm.nnnorm y) 1 → LE. …
          r : NNReal
          hr : LT.lt r (HMul.hMul (NNNorm.nnnorm a) (NNNorm.nnnorm a))
          hr' : LT.lt (NNReal.sqrt r) (SupSet.sSup (Set.image (fun x => NNNorm.nnnorm (a …
          x : A
          hx : Membership.mem (Metric.closedBall 0 1) x
          hxr : LT.lt (NNReal.sqrt r) ((fun x => NNNorm.nnnorm (a.toProd.1 x)) x)
          hx' : LE.le (NNNorm.nnnorm x) 1
          ⊢ Exists fun a_1 => And (Membership.mem (Metric.closedBall 0 1) a_1) (LT.lt r  …
        -/
        refine ⟨star x, mem_closedBall_zero_iff.2 ((nnnorm_star x).trans_le hx'), ?_⟩
        /-
          case refine_2.intro.intro.intro.intro
          𝕜 : Type u_1
          A : Type u_2
          inst✝⁸ : DenselyNormedField 𝕜
          inst✝⁷ : StarRing 𝕜
          inst✝⁶ : NonUnitalNormedRing A
          inst✝⁵ : StarRing A
          inst✝⁴ : CStarRing A
          inst✝³ : NormedSpace 𝕜 A
          inst✝² : SMulCommClass 𝕜 A A
          inst✝¹ : IsScalarTower 𝕜 A A
          inst✝ : StarModule 𝕜 A
          a : DoubleCentralizer 𝕜 A
          hball : (Metric.closedBall 0 1).Nonempty
          key : ∀ (x y : A), LE.le (NNNorm.nnnorm x) 1 → LE.le (NNNorm.nnnorm y) 1 → LE. …
          r : NNReal
          hr : LT.lt r (HMul.hMul (NNNorm.nnnorm a) (NNNorm.nnnorm a))
          hr' : LT.lt (NNReal.sqrt r) (SupSet.sSup (Set.image (fun x => NNNorm.nnnorm (a …
          x : A
          hx : Membership.mem (Metric.closedBall 0 1) x
          hxr : LT.lt (NNReal.sqrt r) ((fun x => NNNorm.nnnorm (a.toProd.1 x)) x)
          hx' : LE.le (NNNorm.nnnorm x) 1
          ⊢ LT.lt r (SupSet.sSup (Set.image (fun a_1 => NNNorm.nnnorm (HMul.hMul (a.toPr …
        -/
        refine lt_csSup_of_lt ?_ ⟨x, hx, rfl⟩ ?_
          /-
            case refine_2.intro.intro.intro.intro.refine_1
            𝕜 : Type u_1
            A : Type u_2
            inst✝⁸ : DenselyNormedField 𝕜
            inst✝⁷ : StarRing 𝕜
            inst✝⁶ : NonUnitalNormedRing A
            inst✝⁵ : StarRing A
            inst✝⁴ : CStarRing A
            inst✝³ : NormedSpace 𝕜 A
            inst✝² : SMulCommClass 𝕜 A A
            inst✝¹ : IsScalarTower 𝕜 A A
            inst✝ : StarModule 𝕜 A
            a : DoubleCentralizer 𝕜 A
            hball : (Metric.closedBall 0 1).Nonempty
            key : ∀ (x y : A), LE.le (NNNorm.nnnorm x) 1 → LE.le (NNNorm.nnnorm y) 1 → LE. …
            r : NNReal
            hr : LT.lt r (HMul.hMul (NNNorm.nnnorm a) (NNNorm.nnnorm a))
            hr' : LT.lt (NNReal.sqrt r) (SupSet.sSup (Set.image (fun x => NNNorm.nnnorm (a …
            x : A
            hx : Membership.mem (Metric.closedBall 0 1) x
            hxr : LT.lt (NNReal.sqrt r) ((fun x => NNNorm.nnnorm (a.toProd.1 x)) x)
            hx' : LE.le (NNNorm.nnnorm x) 1
            ⊢ BddAbove (Set.image (fun a_1 => NNNorm.nnnorm (HMul.hMul (a.toProd.2 (Star.s …
          -/
        · refine ⟨‖a‖₊ * ‖a‖₊, ?_⟩
          /-
            case refine_2.intro.intro.intro.intro.refine_1
            𝕜 : Type u_1
            A : Type u_2
            inst✝⁸ : DenselyNormedField 𝕜
            inst✝⁷ : StarRing 𝕜
            inst✝⁶ : NonUnitalNormedRing A
            inst✝⁵ : StarRing A
            inst✝⁴ : CStarRing A
            inst✝³ : NormedSpace 𝕜 A
            inst✝² : SMulCommClass 𝕜 A A
            inst✝¹ : IsScalarTower 𝕜 A A
            inst✝ : StarModule 𝕜 A
            a : DoubleCentralizer 𝕜 A
            hball : (Metric.closedBall 0 1).Nonempty
            key : ∀ (x y : A), LE.le (NNNorm.nnnorm x) 1 → LE.le (NNNorm.nnnorm y) 1 → LE. …
            r : NNReal
            hr : LT.lt r (HMul.hMul (NNNorm.nnnorm a) (NNNorm.nnnorm a))
            hr' : LT.lt (NNReal.sqrt r) (SupSet.sSup (Set.image (fun x => NNNorm.nnnorm (a …
            x : A
            hx : Membership.mem (Metric.closedBall 0 1) x
            hxr : LT.lt (NNReal.sqrt r) ((fun x => NNNorm.nnnorm (a.toProd.1 x)) x)
            hx' : LE.le (NNNorm.nnnorm x) 1
            ⊢ Membership.mem (upperBounds (Set.image (fun a_1 => NNNorm.nnnorm (HMul.hMul  …
          -/
          rintro - ⟨y, hy, rfl⟩
          /-
            case refine_2.intro.intro.intro.intro.refine_1.intro.intro
            𝕜 : Type u_1
            A : Type u_2
            inst✝⁸ : DenselyNormedField 𝕜
            inst✝⁷ : StarRing 𝕜
            inst✝⁶ : NonUnitalNormedRing A
            inst✝⁵ : StarRing A
            inst✝⁴ : CStarRing A
            inst✝³ : NormedSpace 𝕜 A
            inst✝² : SMulCommClass 𝕜 A A
            inst✝¹ : IsScalarTower 𝕜 A A
            inst✝ : StarModule 𝕜 A
            a : DoubleCentralizer 𝕜 A
            hball : (Metric.closedBall 0 1).Nonempty
            key : ∀ (x y : A), LE.le (NNNorm.nnnorm x) 1 → LE.le (NNNorm.nnnorm y) 1 → LE. …
            r : NNReal
            hr : LT.lt r (HMul.hMul (NNNorm.nnnorm a) (NNNorm.nnnorm a))
            hr' : LT.lt (NNReal.sqrt r) (SupSet.sSup (Set.image (fun x => NNNorm.nnnorm (a …
            x : A
            hx : Membership.mem (Metric.closedBall 0 1) x
            hxr : LT.lt (NNReal.sqrt r) ((fun x => NNNorm.nnnorm (a.toProd.1 x)) x)
            hx' : LE.le (NNNorm.nnnorm x) 1
            y : A
            hy : Membership.mem (Metric.closedBall 0 1) y
            ⊢ LE.le ((fun a_1 => NNNorm.nnnorm (HMul.hMul (a.toProd.2 (Star.star (a.toProd …
          -/
          exact key (star x) y ((nnnorm_star x).trans_le hx') (mem_closedBall_zero_iff.1 hy)
          /-
            🎉 no goals
          -/
        · simpa only [a.central, star_star, CStarRing.nnnorm_star_mul_self, NNReal.sq_sqrt, ← sq]
            using pow_lt_pow_left₀ hxr zero_le' two_ne_zero


noncomputable instance {A : Type*} [NonUnitalCStarAlgebra A] : CStarAlgebra 𝓜(ℂ, A) where


