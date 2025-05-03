/-- The product of all `adicCompletionIntegers`, where `v` runs over the maximal ideals of `R`. -/
def FiniteIntegralAdeles : Type _ :=
  ∀ v : HeightOneSpectrum R, v.adicCompletionIntegers K
-- deriving CommRing, TopologicalSpace, Inhabited

-- Porting note(https://github.com/leanprover-community/mathlib4/issues/5020): added

instance : CommRing (FiniteIntegralAdeles R K) :=
  inferInstanceAs (CommRing (∀ v : HeightOneSpectrum R, v.adicCompletionIntegers K))


instance : TopologicalSpace (FiniteIntegralAdeles R K) :=
  inferInstanceAs (TopologicalSpace (∀ v : HeightOneSpectrum R, v.adicCompletionIntegers K))


instance (v : HeightOneSpectrum R) : TopologicalRing (v.adicCompletionIntegers K) :=
  Subring.instTopologicalRing ..


instance : TopologicalRing (FiniteIntegralAdeles R K) :=
  inferInstanceAs (TopologicalRing (∀ v : HeightOneSpectrum R, v.adicCompletionIntegers K))


instance : Inhabited (FiniteIntegralAdeles R K) :=
  inferInstanceAs (Inhabited (∀ v : HeightOneSpectrum R, v.adicCompletionIntegers K))


local notation "R_hat" => FiniteIntegralAdeles


/-- The product of all `adicCompletion`, where `v` runs over the maximal ideals of `R`. -/
def ProdAdicCompletions :=
  ∀ v : HeightOneSpectrum R, v.adicCompletion K
-- deriving NonUnitalNonAssocRing, TopologicalSpace, TopologicalRing, CommRing, Inhabited


instance : NonUnitalNonAssocRing (ProdAdicCompletions R K) :=
  inferInstanceAs (NonUnitalNonAssocRing (∀ v : HeightOneSpectrum R, v.adicCompletion K))


instance : TopologicalSpace (ProdAdicCompletions R K) :=
  inferInstanceAs (TopologicalSpace (∀ v : HeightOneSpectrum R, v.adicCompletion K))


instance : TopologicalRing (ProdAdicCompletions R K) :=
  inferInstanceAs (TopologicalRing (∀ v : HeightOneSpectrum R, v.adicCompletion K))


instance : CommRing (ProdAdicCompletions R K) :=
  inferInstanceAs (CommRing (∀ v : HeightOneSpectrum R, v.adicCompletion K))


instance : Inhabited (ProdAdicCompletions R K) :=
  inferInstanceAs (Inhabited (∀ v : HeightOneSpectrum R, v.adicCompletion K))


local notation "K_hat" => ProdAdicCompletions


noncomputable instance : Coe (R_hat R K) (K_hat R K) where coe x v := x v


theorem coe_apply (x : R_hat R K) (v : HeightOneSpectrum R) : (x : K_hat R K) v = ↑(x v) :=
  rfl


/-- The inclusion of `R_hat` in `K_hat` as a homomorphism of additive monoids. -/
@[simps]
def Coe.addMonoidHom : AddMonoidHom (R_hat R K) (K_hat R K) where
  toFun := (↑)
  map_zero' := rfl
  map_add' x y := by
    -- Porting note (https://github.com/leanprover-community/mathlib4/issues/11041): was `ext v`
    /-
      R : Type u_1
      K : Type u_2
      inst✝⁴ : CommRing R
      inst✝³ : IsDedekindDomain R
      inst✝² : Field K
      inst✝¹ : Algebra R K
      inst✝ : IsFractionRing R K
      v : IsDedekindDomain.HeightOneSpectrum R
      x y : DedekindDomain.FiniteIntegralAdeles R K
      ⊢ Eq ({ toFun := fun x v => ↑(x v), map_zero' := ⋯ }.toFun (HAdd.hAdd x y)) (H …
    -/
    refine funext fun v => ?_
    /-
      R : Type u_1
      K : Type u_2
      inst✝⁴ : CommRing R
      inst✝³ : IsDedekindDomain R
      inst✝² : Field K
      inst✝¹ : Algebra R K
      inst✝ : IsFractionRing R K
      v✝ : IsDedekindDomain.HeightOneSpectrum R
      x y : DedekindDomain.FiniteIntegralAdeles R K
      v : IsDedekindDomain.HeightOneSpectrum R
      ⊢ Eq ({ toFun := fun x v => ↑(x v), map_zero' := ⋯ }.toFun (HAdd.hAdd x y) v)  …
    -/
    simp only [coe_apply, Pi.add_apply, Subring.coe_add]
    -- Porting note: added
    /-
      R : Type u_1
      K : Type u_2
      inst✝⁴ : CommRing R
      inst✝³ : IsDedekindDomain R
      inst✝² : Field K
      inst✝¹ : Algebra R K
      inst✝ : IsFractionRing R K
      v✝ : IsDedekindDomain.HeightOneSpectrum R
      x y : DedekindDomain.FiniteIntegralAdeles R K
      v : IsDedekindDomain.HeightOneSpectrum R
      ⊢ Eq (↑(HAdd.hAdd x y v)) (HAdd.hAdd (fun v => ↑(x v)) (fun v => ↑(y v)) v)
    -/
    rw [Pi.add_apply, Pi.add_apply, Subring.coe_add]
    /-
      🎉 no goals
    -/


/-- The inclusion of `R_hat` in `K_hat` as a ring homomorphism. -/
@[simps]
def Coe.ringHom : RingHom (R_hat R K) (K_hat R K) :=
  { Coe.addMonoidHom R K with
    toFun := (↑)
    map_one' := rfl
    map_mul' := fun x y => by
      -- Porting note (https://github.com/leanprover-community/mathlib4/issues/11041): was `ext p`
      /-
        R : Type u_1
        K : Type u_2
        inst✝⁴ : CommRing R
        inst✝³ : IsDedekindDomain R
        inst✝² : Field K
        inst✝¹ : Algebra R K
        inst✝ : IsFractionRing R K
        v : IsDedekindDomain.HeightOneSpectrum R
        x y : DedekindDomain.FiniteIntegralAdeles R K
        ⊢ Eq ({ toFun := fun x v => ↑(x v), map_one' := ⋯ }.toFun (HMul.hMul x y)) (HM …
      -/
      refine funext fun p => ?_
      /-
        R : Type u_1
        K : Type u_2
        inst✝⁴ : CommRing R
        inst✝³ : IsDedekindDomain R
        inst✝² : Field K
        inst✝¹ : Algebra R K
        inst✝ : IsFractionRing R K
        v : IsDedekindDomain.HeightOneSpectrum R
        x y : DedekindDomain.FiniteIntegralAdeles R K
        p : IsDedekindDomain.HeightOneSpectrum R
        ⊢ Eq ({ toFun := fun x v => ↑(x v), map_one' := ⋯ }.toFun (HMul.hMul x y) p) ( …
      -/
      simp only [Pi.mul_apply, Subring.coe_mul]
      -- Porting note: added
      /-
        R : Type u_1
        K : Type u_2
        inst✝⁴ : CommRing R
        inst✝³ : IsDedekindDomain R
        inst✝² : Field K
        inst✝¹ : Algebra R K
        inst✝ : IsFractionRing R K
        v : IsDedekindDomain.HeightOneSpectrum R
        x y : DedekindDomain.FiniteIntegralAdeles R K
        p : IsDedekindDomain.HeightOneSpectrum R
        ⊢ Eq (↑(HMul.hMul x y p)) (HMul.hMul (fun v => ↑(x v)) (fun v => ↑(y v)) p)
      -/
      rw [Pi.mul_apply, Pi.mul_apply, Subring.coe_mul] }
      /-
        🎉 no goals
      -/


instance : Algebra K (K_hat R K) :=
      /-
        R : Type u_1
        K : Type u_2
        inst✝⁴ : CommRing R
        inst✝³ : IsDedekindDomain R
        inst✝² : Field K
        inst✝¹ : Algebra R K
        inst✝ : IsFractionRing R K
        v : IsDedekindDomain.HeightOneSpectrum R
        ⊢ Algebra K ((v : IsDedekindDomain.HeightOneSpectrum R) → IsDedekindDomain.Hei …
      -/
  (by infer_instance : Algebra K <| ∀ v : HeightOneSpectrum R, v.adicCompletion K)
      /-
        🎉 no goals
      -/


@[simp]
lemma ProdAdicCompletions.algebraMap_apply' (k : K) :
    algebraMap K (K_hat R K) k v = (k : v.adicCompletion K) := rfl


instance ProdAdicCompletions.algebra' : Algebra R (K_hat R K) :=
      /-
        R : Type u_1
        K : Type u_2
        inst✝⁴ : CommRing R
        inst✝³ : IsDedekindDomain R
        inst✝² : Field K
        inst✝¹ : Algebra R K
        inst✝ : IsFractionRing R K
        v : IsDedekindDomain.HeightOneSpectrum R
        ⊢ Algebra R ((v : IsDedekindDomain.HeightOneSpectrum R) → IsDedekindDomain.Hei …
      -/
  (by infer_instance : Algebra R <| ∀ v : HeightOneSpectrum R, v.adicCompletion K)
      /-
        🎉 no goals
      -/


@[simp]
lemma ProdAdicCompletions.algebraMap_apply (r : R) :
    algebraMap R (K_hat R K) r v = (algebraMap R K r : v.adicCompletion K) := rfl


instance : IsScalarTower R K (K_hat R K) :=
      /-
        R : Type u_1
        K : Type u_2
        inst✝⁴ : CommRing R
        inst✝³ : IsDedekindDomain R
        inst✝² : Field K
        inst✝¹ : Algebra R K
        inst✝ : IsFractionRing R K
        v : IsDedekindDomain.HeightOneSpectrum R
        ⊢ IsScalarTower R K ((v : IsDedekindDomain.HeightOneSpectrum R) → IsDedekindDo …
      -/
  (by infer_instance : IsScalarTower R K <| ∀ v : HeightOneSpectrum R, v.adicCompletion K)
      /-
        🎉 no goals
      -/


instance : Algebra R (R_hat R K) :=
      /-
        R : Type u_1
        K : Type u_2
        inst✝⁴ : CommRing R
        inst✝³ : IsDedekindDomain R
        inst✝² : Field K
        inst✝¹ : Algebra R K
        inst✝ : IsFractionRing R K
        v : IsDedekindDomain.HeightOneSpectrum R
        ⊢ Algebra R ((v : IsDedekindDomain.HeightOneSpectrum R) → Subtype fun x => Mem …
      -/
  (by infer_instance : Algebra R <| ∀ v : HeightOneSpectrum R, v.adicCompletionIntegers K)
      /-
        🎉 no goals
      -/


instance ProdAdicCompletions.algebraCompletions : Algebra (R_hat R K) (K_hat R K) :=
  (FiniteIntegralAdeles.Coe.ringHom R K).toAlgebra


instance ProdAdicCompletions.isScalarTower_completions : IsScalarTower R (R_hat R K) (K_hat R K) :=
      /-
        R : Type u_1
        K : Type u_2
        inst✝⁴ : CommRing R
        inst✝³ : IsDedekindDomain R
        inst✝² : Field K
        inst✝¹ : Algebra R K
        inst✝ : IsFractionRing R K
        v : IsDedekindDomain.HeightOneSpectrum R
        ⊢ IsScalarTower R ((v : IsDedekindDomain.HeightOneSpectrum R) → Subtype fun x  …
      -/
  (by infer_instance :
      /-
        🎉 no goals
      -/
    IsScalarTower R (∀ v : HeightOneSpectrum R, v.adicCompletionIntegers K) <|
      ∀ v : HeightOneSpectrum R, v.adicCompletion K)


/-- The inclusion of `R_hat` in `K_hat` as an algebra homomorphism. -/
def Coe.algHom : AlgHom R (R_hat R K) (K_hat R K) :=
  { Coe.ringHom R K with
    toFun := (↑)
    commutes' := fun _ => rfl }


theorem Coe.algHom_apply (x : R_hat R K) (v : HeightOneSpectrum R) : (Coe.algHom R K) x v = x v :=
  rfl


/-- An element `x : K_hat R K` is a finite adèle if for all but finitely many height one ideals
  `v`, the component `x v` is a `v`-adic integer. -/
def IsFiniteAdele (x : K_hat R K) :=
  ∀ᶠ v : HeightOneSpectrum R in Filter.cofinite, x v ∈ v.adicCompletionIntegers K


@[simp]
lemma isFiniteAdele_iff (x : K_hat R K) :
    x.IsFiniteAdele ↔ {v | x v ∉ adicCompletionIntegers K v}.Finite := Iff.rfl


/-- The sum of two finite adèles is a finite adèle. -/
theorem add {x y : K_hat R K} (hx : x.IsFiniteAdele) (hy : y.IsFiniteAdele) :
    (x + y).IsFiniteAdele := by
  /-
    R : Type u_1
    K : Type u_2
    inst✝⁴ : CommRing R
    inst✝³ : IsDedekindDomain R
    inst✝² : Field K
    inst✝¹ : Algebra R K
    inst✝ : IsFractionRing R K
    x y : DedekindDomain.ProdAdicCompletions R K
    hx : x.IsFiniteAdele
    hy : y.IsFiniteAdele
    ⊢ (HAdd.hAdd x y).IsFiniteAdele
  -/
  rw [IsFiniteAdele, Filter.eventually_cofinite] at hx hy ⊢
  have h_subset :
    {v : HeightOneSpectrum R | ¬(x + y) v ∈ v.adicCompletionIntegers K} ⊆
      {v : HeightOneSpectrum R | ¬x v ∈ v.adicCompletionIntegers K} ∪
        {v : HeightOneSpectrum R | ¬y v ∈ v.adicCompletionIntegers K} := by
    intro v hv
    rw [mem_union, mem_setOf, mem_setOf]
    rw [mem_setOf] at hv
    contrapose! hv
    rw [mem_adicCompletionIntegers, mem_adicCompletionIntegers, ← max_le_iff] at hv
    rw [mem_adicCompletionIntegers, Pi.add_apply]
    exact le_trans (Valued.v.map_add_le_max' (x v) (y v)) hv
  /-
    R : Type u_1
    K : Type u_2
    inst✝⁴ : CommRing R
    inst✝³ : IsDedekindDomain R
    inst✝² : Field K
    inst✝¹ : Algebra R K
    inst✝ : IsFractionRing R K
    x y : DedekindDomain.ProdAdicCompletions R K
    hx : (setOf fun x_1 => Not (Membership.mem (IsDedekindDomain.HeightOneSpectrum …
    hy : (setOf fun x => Not (Membership.mem (IsDedekindDomain.HeightOneSpectrum.a …
    h_subset : HasSubset.Subset (setOf fun v => Not (Membership.mem (IsDedekindDom …
    ⊢ (setOf fun x_1 => Not (Membership.mem (IsDedekindDomain.HeightOneSpectrum.ad …
  -/
  exact (hx.union hy).subset h_subset
  /-
    🎉 no goals
  -/


/-- The tuple `(0)_v` is a finite adèle. -/
theorem zero : (0 : K_hat R K).IsFiniteAdele := by
  /-
    R : Type u_1
    K : Type u_2
    inst✝⁴ : CommRing R
    inst✝³ : IsDedekindDomain R
    inst✝² : Field K
    inst✝¹ : Algebra R K
    inst✝ : IsFractionRing R K
    ⊢ DedekindDomain.ProdAdicCompletions.IsFiniteAdele 0
  -/
  rw [IsFiniteAdele, Filter.eventually_cofinite]
  have h_empty :
    {v : HeightOneSpectrum R | ¬(0 : v.adicCompletion K) ∈ v.adicCompletionIntegers K} = ∅ := by
    ext v; rw [mem_empty_iff_false, iff_false]; intro hv
    rw [mem_setOf] at hv; apply hv; rw [mem_adicCompletionIntegers]
    have h_zero : (Valued.v (0 : v.adicCompletion K) : WithZero (Multiplicative ℤ)) = 0 :=
      Valued.v.map_zero'
    rw [h_zero]; exact zero_le_one' _
  -- Porting note: was `exact`, but `OfNat` got in the way.
  /-
    R : Type u_1
    K : Type u_2
    inst✝⁴ : CommRing R
    inst✝³ : IsDedekindDomain R
    inst✝² : Field K
    inst✝¹ : Algebra R K
    inst✝ : IsFractionRing R K
    h_empty : Eq (setOf fun v => Not (Membership.mem (IsDedekindDomain.HeightOneSp …
    ⊢ (setOf fun x => Not (Membership.mem (IsDedekindDomain.HeightOneSpectrum.adic …
  -/
  convert finite_empty
  /-
    🎉 no goals
  -/


/-- The negative of a finite adèle is a finite adèle. -/
theorem neg {x : K_hat R K} (hx : x.IsFiniteAdele) : (-x).IsFiniteAdele := by
  /-
    R : Type u_1
    K : Type u_2
    inst✝⁴ : CommRing R
    inst✝³ : IsDedekindDomain R
    inst✝² : Field K
    inst✝¹ : Algebra R K
    inst✝ : IsFractionRing R K
    x : DedekindDomain.ProdAdicCompletions R K
    hx : x.IsFiniteAdele
    ⊢ (Neg.neg x).IsFiniteAdele
  -/
  rw [IsFiniteAdele] at hx ⊢
  have h :
    ∀ v : HeightOneSpectrum R,
      -x v ∈ v.adicCompletionIntegers K ↔ x v ∈ v.adicCompletionIntegers K := by
    intro v
    rw [mem_adicCompletionIntegers, mem_adicCompletionIntegers, Valuation.map_neg]
  -- Porting note: was `simpa only [Pi.neg_apply, h] using hx` but `Pi.neg_apply` no longer works
  /-
    R : Type u_1
    K : Type u_2
    inst✝⁴ : CommRing R
    inst✝³ : IsDedekindDomain R
    inst✝² : Field K
    inst✝¹ : Algebra R K
    inst✝ : IsFractionRing R K
    x : DedekindDomain.ProdAdicCompletions R K
    hx : Filter.Eventually (fun v => Membership.mem (IsDedekindDomain.HeightOneSpe …
    h : ∀ (v : IsDedekindDomain.HeightOneSpectrum R), Iff (Membership.mem (IsDedek …
    ⊢ Filter.Eventually (fun v => Membership.mem (IsDedekindDomain.HeightOneSpectr …
  -/
  convert hx using 2 with v
  /-
    case h.e'_2.h.a
    R : Type u_1
    K : Type u_2
    inst✝⁴ : CommRing R
    inst✝³ : IsDedekindDomain R
    inst✝² : Field K
    inst✝¹ : Algebra R K
    inst✝ : IsFractionRing R K
    x : DedekindDomain.ProdAdicCompletions R K
    hx : Filter.Eventually (fun v => Membership.mem (IsDedekindDomain.HeightOneSpe …
    h : ∀ (v : IsDedekindDomain.HeightOneSpectrum R), Iff (Membership.mem (IsDedek …
    v : IsDedekindDomain.HeightOneSpectrum R
    ⊢ Iff (Membership.mem (IsDedekindDomain.HeightOneSpectrum.adicCompletionIntege …
  -/
  convert h v
  /-
    🎉 no goals
  -/


/-- The product of two finite adèles is a finite adèle. -/
theorem mul {x y : K_hat R K} (hx : x.IsFiniteAdele) (hy : y.IsFiniteAdele) :
    (x * y).IsFiniteAdele := by
  /-
    R : Type u_1
    K : Type u_2
    inst✝⁴ : CommRing R
    inst✝³ : IsDedekindDomain R
    inst✝² : Field K
    inst✝¹ : Algebra R K
    inst✝ : IsFractionRing R K
    x y : DedekindDomain.ProdAdicCompletions R K
    hx : x.IsFiniteAdele
    hy : y.IsFiniteAdele
    ⊢ (HMul.hMul x y).IsFiniteAdele
  -/
  rw [IsFiniteAdele, Filter.eventually_cofinite] at hx hy ⊢
  have h_subset :
    {v : HeightOneSpectrum R | ¬(x * y) v ∈ v.adicCompletionIntegers K} ⊆
      {v : HeightOneSpectrum R | ¬x v ∈ v.adicCompletionIntegers K} ∪
        {v : HeightOneSpectrum R | ¬y v ∈ v.adicCompletionIntegers K} := by
    intro v hv
    rw [mem_union, mem_setOf, mem_setOf]
    rw [mem_setOf] at hv
    contrapose! hv
    rw [mem_adicCompletionIntegers, mem_adicCompletionIntegers] at hv
    have h_mul : Valued.v (x v * y v) = Valued.v (x v) * Valued.v (y v) :=
      Valued.v.map_mul' (x v) (y v)
    rw [mem_adicCompletionIntegers, Pi.mul_apply, h_mul]
    exact mul_le_one' hv.left hv.right
  /-
    R : Type u_1
    K : Type u_2
    inst✝⁴ : CommRing R
    inst✝³ : IsDedekindDomain R
    inst✝² : Field K
    inst✝¹ : Algebra R K
    inst✝ : IsFractionRing R K
    x y : DedekindDomain.ProdAdicCompletions R K
    hx : (setOf fun x_1 => Not (Membership.mem (IsDedekindDomain.HeightOneSpectrum …
    hy : (setOf fun x => Not (Membership.mem (IsDedekindDomain.HeightOneSpectrum.a …
    h_subset : HasSubset.Subset (setOf fun v => Not (Membership.mem (IsDedekindDom …
    ⊢ (setOf fun x_1 => Not (Membership.mem (IsDedekindDomain.HeightOneSpectrum.ad …
  -/
  exact (hx.union hy).subset h_subset
  /-
    🎉 no goals
  -/


/-- The tuple `(1)_v` is a finite adèle. -/
theorem one : (1 : K_hat R K).IsFiniteAdele := by
  /-
    R : Type u_1
    K : Type u_2
    inst✝⁴ : CommRing R
    inst✝³ : IsDedekindDomain R
    inst✝² : Field K
    inst✝¹ : Algebra R K
    inst✝ : IsFractionRing R K
    ⊢ DedekindDomain.ProdAdicCompletions.IsFiniteAdele 1
  -/
  rw [IsFiniteAdele, Filter.eventually_cofinite]
  have h_empty :
    {v : HeightOneSpectrum R | ¬(1 : v.adicCompletion K) ∈ v.adicCompletionIntegers K} = ∅ := by
    ext v; rw [mem_empty_iff_false, iff_false]; intro hv
    rw [mem_setOf] at hv; apply hv; rw [mem_adicCompletionIntegers]
    exact le_of_eq Valued.v.map_one'
  -- Porting note: was `exact`, but `OfNat` got in the way.
  /-
    R : Type u_1
    K : Type u_2
    inst✝⁴ : CommRing R
    inst✝³ : IsDedekindDomain R
    inst✝² : Field K
    inst✝¹ : Algebra R K
    inst✝ : IsFractionRing R K
    h_empty : Eq (setOf fun v => Not (Membership.mem (IsDedekindDomain.HeightOneSp …
    ⊢ (setOf fun x => Not (Membership.mem (IsDedekindDomain.HeightOneSpectrum.adic …
  -/
  convert finite_empty
  /-
    🎉 no goals
  -/


theorem algebraMap' (k : K) : (_root_.algebraMap K (K_hat R K) k).IsFiniteAdele := by
  /-
    R : Type u_1
    K : Type u_2
    inst✝⁴ : CommRing R
    inst✝³ : IsDedekindDomain R
    inst✝² : Field K
    inst✝¹ : Algebra R K
    inst✝ : IsFractionRing R K
    k : K
    ⊢ ((algebraMap K (DedekindDomain.ProdAdicCompletions R K)) k).IsFiniteAdele
  -/
  rw [IsFiniteAdele, Filter.eventually_cofinite]
  simp_rw [mem_adicCompletionIntegers, ProdAdicCompletions.algebraMap_apply',
    Valued.valuedCompletion_apply, not_le]
  /-
    R : Type u_1
    K : Type u_2
    inst✝⁴ : CommRing R
    inst✝³ : IsDedekindDomain R
    inst✝² : Field K
    inst✝¹ : Algebra R K
    inst✝ : IsFractionRing R K
    k : K
    ⊢ (setOf fun x => LT.lt 1 (Valued.v k)).Finite
  -/
  change {v : HeightOneSpectrum R | 1 < v.valuation k}.Finite
  -- The goal currently: if k ∈ K = field of fractions of a Dedekind domain R,
  -- then v(k)>1 for only finitely many v.
  -- We now write k=n/d and go via R to solve this goal. Do we need to do this?
  /-
    R : Type u_1
    K : Type u_2
    inst✝⁴ : CommRing R
    inst✝³ : IsDedekindDomain R
    inst✝² : Field K
    inst✝¹ : Algebra R K
    inst✝ : IsFractionRing R K
    k : K
    ⊢ (setOf fun v => LT.lt 1 (v.valuation k)).Finite
  -/
  obtain ⟨⟨n, ⟨d, hd⟩⟩, hk⟩ := IsLocalization.surj (nonZeroDivisors R) k
  /-
    case intro.mk.mk
    R : Type u_1
    K : Type u_2
    inst✝⁴ : CommRing R
    inst✝³ : IsDedekindDomain R
    inst✝² : Field K
    inst✝¹ : Algebra R K
    inst✝ : IsFractionRing R K
    k : K
    n d : R
    hd : Membership.mem (nonZeroDivisors R) d
    hk : Eq (HMul.hMul k ((algebraMap R K) ↑{ fst := n, snd := ⟨d, hd⟩ }.2)) ((alg …
    ⊢ (setOf fun v => LT.lt 1 (v.valuation k)).Finite
  -/
  have hd' : d ≠ 0 := nonZeroDivisors.ne_zero hd
  suffices {v : HeightOneSpectrum R | v.valuation (_root_.algebraMap R K d : K) < 1}.Finite by
    apply Finite.subset this
    intro v hv
    apply_fun v.valuation at hk
    simp only [Valuation.map_mul, valuation_of_algebraMap] at hk
    rw [mem_setOf_eq, valuation_of_algebraMap]
    have := intValuation_le_one v n
    contrapose! this
    change 1 < v.intValuation n
    rw [← hk, mul_comm]
    exact lt_mul_of_le_of_one_lt' this hv (by simp) (by simp)
  /-
    case intro.mk.mk
    R : Type u_1
    K : Type u_2
    inst✝⁴ : CommRing R
    inst✝³ : IsDedekindDomain R
    inst✝² : Field K
    inst✝¹ : Algebra R K
    inst✝ : IsFractionRing R K
    k : K
    n d : R
    hd : Membership.mem (nonZeroDivisors R) d
    hk : Eq (HMul.hMul k ((algebraMap R K) ↑{ fst := n, snd := ⟨d, hd⟩ }.2)) ((alg …
    hd' : Ne d 0
    ⊢ (setOf fun v => LT.lt (v.valuation ((algebraMap R K) d)) 1).Finite
  -/
  simp_rw [valuation_of_algebraMap]
  /-
    case intro.mk.mk
    R : Type u_1
    K : Type u_2
    inst✝⁴ : CommRing R
    inst✝³ : IsDedekindDomain R
    inst✝² : Field K
    inst✝¹ : Algebra R K
    inst✝ : IsFractionRing R K
    k : K
    n d : R
    hd : Membership.mem (nonZeroDivisors R) d
    hk : Eq (HMul.hMul k ((algebraMap R K) ↑{ fst := n, snd := ⟨d, hd⟩ }.2)) ((alg …
    hd' : Ne d 0
    ⊢ (setOf fun v => LT.lt (v.intValuation d) 1).Finite
  -/
  change {v : HeightOneSpectrum R | v.intValuationDef d < 1}.Finite
  /-
    case intro.mk.mk
    R : Type u_1
    K : Type u_2
    inst✝⁴ : CommRing R
    inst✝³ : IsDedekindDomain R
    inst✝² : Field K
    inst✝¹ : Algebra R K
    inst✝ : IsFractionRing R K
    k : K
    n d : R
    hd : Membership.mem (nonZeroDivisors R) d
    hk : Eq (HMul.hMul k ((algebraMap R K) ↑{ fst := n, snd := ⟨d, hd⟩ }.2)) ((alg …
    hd' : Ne d 0
    ⊢ (setOf fun v => LT.lt (v.intValuationDef d) 1).Finite
  -/
  simp_rw [intValuation_lt_one_iff_dvd]
  /-
    case intro.mk.mk
    R : Type u_1
    K : Type u_2
    inst✝⁴ : CommRing R
    inst✝³ : IsDedekindDomain R
    inst✝² : Field K
    inst✝¹ : Algebra R K
    inst✝ : IsFractionRing R K
    k : K
    n d : R
    hd : Membership.mem (nonZeroDivisors R) d
    hk : Eq (HMul.hMul k ((algebraMap R K) ↑{ fst := n, snd := ⟨d, hd⟩ }.2)) ((alg …
    hd' : Ne d 0
    ⊢ (setOf fun v => Dvd.dvd v.asIdeal (Ideal.span (Singleton.singleton d))).Finite
  -/
  apply Ideal.finite_factors
  /-
    case intro.mk.mk.hI
    R : Type u_1
    K : Type u_2
    inst✝⁴ : CommRing R
    inst✝³ : IsDedekindDomain R
    inst✝² : Field K
    inst✝¹ : Algebra R K
    inst✝ : IsFractionRing R K
    k : K
    n d : R
    hd : Membership.mem (nonZeroDivisors R) d
    hk : Eq (HMul.hMul k ((algebraMap R K) ↑{ fst := n, snd := ⟨d, hd⟩ }.2)) ((alg …
    hd' : Ne d 0
    ⊢ Ne (Ideal.span (Singleton.singleton d)) 0
  -/
  simpa only [Submodule.zero_eq_bot, ne_eq, Ideal.span_singleton_eq_bot]
  /-
    🎉 no goals
  -/


/-- The finite adèle ring of `R` is the restricted product over all maximal ideals `v` of `R`
of `adicCompletion`, with respect to `adicCompletionIntegers`.

Note that we make this a `Type` rather than a `Subtype` (e.g., a `Subalgebra`) since we wish
to endow it with a finer topology than that of the subspace topology. -/
def FiniteAdeleRing : Type _ := {x : K_hat R K // x.IsFiniteAdele}


/-- The finite adèle ring of `R`, regarded as a `K`-subalgebra of the
product of the local completions of `K`.

Note that this definition exists to streamline the proof that the finite adèles are an algebra
over `K`, rather than to express their relationship to `K_hat R K` which is essentially a
detail of their construction.
-/
def subalgebra : Subalgebra K (K_hat R K) where
  carrier := {x : K_hat R K | x.IsFiniteAdele}
  mul_mem' := mul
  one_mem' := one
  add_mem' := add
  zero_mem' := zero
  algebraMap_mem' := algebraMap'


instance : CommRing (FiniteAdeleRing R K) :=
  Subalgebra.toCommRing (subalgebra R K)


instance : Algebra K (FiniteAdeleRing R K) :=
  Subalgebra.algebra (subalgebra R K)


instance : Algebra R (FiniteAdeleRing R K) :=
  ((algebraMap K (FiniteAdeleRing R K)).comp (algebraMap R K)).toAlgebra


instance : IsScalarTower R K (FiniteAdeleRing R K) :=
  IsScalarTower.of_algebraMap_eq' rfl


instance : Coe (FiniteAdeleRing R K) (K_hat R K) where
  coe x := x.1


@[ext]
lemma ext {a₁ a₂ : FiniteAdeleRing R K} (h : (a₁ : K_hat R K) = a₂) : a₁ = a₂ :=
  Subtype.ext h


/-- The finite adele ring is an algebra over the finite integral adeles. -/
instance : Algebra (R_hat R K) (FiniteAdeleRing R K) where
  smul rhat fadele := ⟨fun v ↦ rhat v * fadele.1 v, Finite.subset fadele.2 <| fun v hv ↦ by
    /-
      R : Type u_1
      K : Type u_2
      inst✝⁴ : CommRing R
      inst✝³ : IsDedekindDomain R
      inst✝² : Field K
      inst✝¹ : Algebra R K
      inst✝ : IsFractionRing R K
      v✝ : IsDedekindDomain.HeightOneSpectrum R
      rhat : DedekindDomain.FiniteIntegralAdeles R K
      fadele : DedekindDomain.FiniteAdeleRing R K
      v : IsDedekindDomain.HeightOneSpectrum R
      hv : Membership.mem (HasCompl.compl (setOf fun x => (fun v => Membership.mem ( …
      ⊢ Membership.mem (HasCompl.compl (setOf fun x => (fun v => Membership.mem (IsD …
    -/
    simp only [mem_adicCompletionIntegers, mem_compl_iff, mem_setOf_eq, map_mul] at hv ⊢
    /-
      R : Type u_1
      K : Type u_2
      inst✝⁴ : CommRing R
      inst✝³ : IsDedekindDomain R
      inst✝² : Field K
      inst✝¹ : Algebra R K
      inst✝ : IsFractionRing R K
      v✝ : IsDedekindDomain.HeightOneSpectrum R
      rhat : DedekindDomain.FiniteIntegralAdeles R K
      fadele : DedekindDomain.FiniteAdeleRing R K
      v : IsDedekindDomain.HeightOneSpectrum R
      hv : Not (LE.le (HMul.hMul (Valued.v ↑(rhat v)) (Valued.v (↑fadele v))) 1)
      ⊢ Not (LE.le (Valued.v (↑fadele v)) 1)
    -/
    exact mt (mul_le_one' (rhat v).2) hv
    /-
      🎉 no goals
    -/
    ⟩
                    /-
                      R : Type u_1
                      K : Type u_2
                      inst✝⁴ : CommRing R
                      inst✝³ : IsDedekindDomain R
                      inst✝² : Field K
                      inst✝¹ : Algebra R K
                      inst✝ : IsFractionRing R K
                      v : IsDedekindDomain.HeightOneSpectrum R
                      r : DedekindDomain.FiniteIntegralAdeles R K
                      ⊢ DedekindDomain.ProdAdicCompletions.IsFiniteAdele fun v => ↑(r v)
                    -/
  toFun r := ⟨r, by simp_all⟩
                    /-
                      🎉 no goals
                    -/
                 /-
                   R : Type u_1
                   K : Type u_2
                   inst✝⁴ : CommRing R
                   inst✝³ : IsDedekindDomain R
                   inst✝² : Field K
                   inst✝¹ : Algebra R K
                   inst✝ : IsFractionRing R K
                   v : IsDedekindDomain.HeightOneSpectrum R
                   ⊢ Eq ((fun r => ⟨fun v => ↑(r v), ⋯⟩) 1) 1
                 -/
  map_one' := by ext; rfl
                      /-
                        🎉 no goals
                      -/
                     /-
                       R : Type u_1
                       K : Type u_2
                       inst✝⁴ : CommRing R
                       inst✝³ : IsDedekindDomain R
                       inst✝² : Field K
                       inst✝¹ : Algebra R K
                       inst✝ : IsFractionRing R K
                       v : IsDedekindDomain.HeightOneSpectrum R
                       x✝¹ x✝ : DedekindDomain.FiniteIntegralAdeles R K
                       ⊢ Eq ({ toFun := fun r => ⟨fun v => ↑(r v), ⋯⟩, map_one' := ⋯ }.toFun (HMul.hM …
                     -/
  map_mul' _ _ := by ext; rfl
                          /-
                            🎉 no goals
                          -/
                  /-
                    R : Type u_1
                    K : Type u_2
                    inst✝⁴ : CommRing R
                    inst✝³ : IsDedekindDomain R
                    inst✝² : Field K
                    inst✝¹ : Algebra R K
                    inst✝ : IsFractionRing R K
                    v : IsDedekindDomain.HeightOneSpectrum R
                    ⊢ Eq ((↑{ toFun := fun r => ⟨fun v => ↑(r v), ⋯⟩, map_one' := ⋯, map_mul' := ⋯ …
                  -/
  map_zero' := by ext; rfl
                       /-
                         🎉 no goals
                       -/
                     /-
                       R : Type u_1
                       K : Type u_2
                       inst✝⁴ : CommRing R
                       inst✝³ : IsDedekindDomain R
                       inst✝² : Field K
                       inst✝¹ : Algebra R K
                       inst✝ : IsFractionRing R K
                       v : IsDedekindDomain.HeightOneSpectrum R
                       x✝¹ x✝ : DedekindDomain.FiniteIntegralAdeles R K
                       ⊢ Eq ((↑{ toFun := fun r => ⟨fun v => ↑(r v), ⋯⟩, map_one' := ⋯, map_mul' := ⋯ …
                     -/
  map_add' _ _ := by ext; rfl
                          /-
                            🎉 no goals
                          -/
  commutes' _ _ := mul_comm _ _
  smul_def' _ _ := rfl


instance : CoeFun (FiniteAdeleRing R K)
    (fun _ ↦ ∀ (v : HeightOneSpectrum R), adicCompletion K v) where
  coe a v := a.1 v


variable {R K} in
lemma exists_finiteIntegralAdele_iff (a : FiniteAdeleRing R K) : (∃ c : R_hat R K,
    a = c) ↔ ∀ (v : HeightOneSpectrum R), a v ∈ adicCompletionIntegers K v :=
      /-
        R : Type u_1
        K : Type u_2
        inst✝⁴ : CommRing R
        inst✝³ : IsDedekindDomain R
        inst✝² : Field K
        inst✝¹ : Algebra R K
        inst✝ : IsFractionRing R K
        a : DedekindDomain.FiniteAdeleRing R K
        ⊢ (Exists fun c => Eq a ↑c) → ∀ (v : IsDedekindDomain.HeightOneSpectrum R), Me …
      -/
  ⟨by rintro ⟨c, rfl⟩ v; exact (c v).2, fun h ↦ ⟨fun v ↦ ⟨a v, h v⟩, rfl⟩⟩
                         /-
                           🎉 no goals
                         -/


variable {R K} in
lemma mul_nonZeroDivisor_mem_finiteIntegralAdeles (a : FiniteAdeleRing R K) :
    ∃ (b : R⁰) (c : R_hat R K), a * ((b : R) : FiniteAdeleRing R K) = c := by
  /-
    R : Type u_1
    K : Type u_2
    inst✝⁴ : CommRing R
    inst✝³ : IsDedekindDomain R
    inst✝² : Field K
    inst✝¹ : Algebra R K
    inst✝ : IsFractionRing R K
    a : DedekindDomain.FiniteAdeleRing R K
    ⊢ Exists fun b => Exists fun c => Eq (HMul.hMul a ↑↑b) ↑c
  -/
  let S := {v | a v ∉ adicCompletionIntegers K v}
  /-
    R : Type u_1
    K : Type u_2
    inst✝⁴ : CommRing R
    inst✝³ : IsDedekindDomain R
    inst✝² : Field K
    inst✝¹ : Algebra R K
    inst✝ : IsFractionRing R K
    a : DedekindDomain.FiniteAdeleRing R K
    S : Set (IsDedekindDomain.HeightOneSpectrum R) := setOf fun v => Not (Membersh …
    ⊢ Exists fun b => Exists fun c => Eq (HMul.hMul a ↑↑b) ↑c
  -/
  choose b hb h using adicCompletion.mul_nonZeroDivisor_mem_adicCompletionIntegers (R := R) (K := K)
  /-
    R : Type u_1
    K : Type u_2
    inst✝⁴ : CommRing R
    inst✝³ : IsDedekindDomain R
    inst✝² : Field K
    inst✝¹ : Algebra R K
    inst✝ : IsFractionRing R K
    a : DedekindDomain.FiniteAdeleRing R K
    S : Set (IsDedekindDomain.HeightOneSpectrum R) := setOf fun v => Not (Membersh …
    b : (v : IsDedekindDomain.HeightOneSpectrum R) → IsDedekindDomain.HeightOneSpe …
    hb : ∀ (v : IsDedekindDomain.HeightOneSpectrum R) (a : IsDedekindDomain.Height …
    h : ∀ (v : IsDedekindDomain.HeightOneSpectrum R) (a : IsDedekindDomain.HeightO …
    ⊢ Exists fun b => Exists fun c => Eq (HMul.hMul a ↑↑b) ↑c
  -/
  let p := ∏ᶠ v ∈ S, b v (a v)
  have hp : p ∈ R⁰ := finprod_mem_induction (· ∈ R⁰) (one_mem _) (fun _ _ => mul_mem) <|
    fun _ _ ↦ hb _ _
  /-
    R : Type u_1
    K : Type u_2
    inst✝⁴ : CommRing R
    inst✝³ : IsDedekindDomain R
    inst✝² : Field K
    inst✝¹ : Algebra R K
    inst✝ : IsFractionRing R K
    a : DedekindDomain.FiniteAdeleRing R K
    S : Set (IsDedekindDomain.HeightOneSpectrum R) := setOf fun v => Not (Membersh …
    b : (v : IsDedekindDomain.HeightOneSpectrum R) → IsDedekindDomain.HeightOneSpe …
    hb : ∀ (v : IsDedekindDomain.HeightOneSpectrum R) (a : IsDedekindDomain.Height …
    h : ∀ (v : IsDedekindDomain.HeightOneSpectrum R) (a : IsDedekindDomain.HeightO …
    p : R := finprod fun v => finprod fun h => b v ((fun v => ↑a v) v)
    hp : Membership.mem (nonZeroDivisors R) p
    ⊢ Exists fun b => Exists fun c => Eq (HMul.hMul a ↑↑b) ↑c
  -/
  use ⟨p, hp⟩
  /-
    case h
    R : Type u_1
    K : Type u_2
    inst✝⁴ : CommRing R
    inst✝³ : IsDedekindDomain R
    inst✝² : Field K
    inst✝¹ : Algebra R K
    inst✝ : IsFractionRing R K
    a : DedekindDomain.FiniteAdeleRing R K
    S : Set (IsDedekindDomain.HeightOneSpectrum R) := setOf fun v => Not (Membersh …
    b : (v : IsDedekindDomain.HeightOneSpectrum R) → IsDedekindDomain.HeightOneSpe …
    hb : ∀ (v : IsDedekindDomain.HeightOneSpectrum R) (a : IsDedekindDomain.Height …
    h : ∀ (v : IsDedekindDomain.HeightOneSpectrum R) (a : IsDedekindDomain.HeightO …
    p : R := finprod fun v => finprod fun h => b v ((fun v => ↑a v) v)
    hp : Membership.mem (nonZeroDivisors R) p
    ⊢ Exists fun c => Eq (HMul.hMul a ↑↑⟨p, hp⟩) ↑c
  -/
  rw [exists_finiteIntegralAdele_iff]
  /-
    case h
    R : Type u_1
    K : Type u_2
    inst✝⁴ : CommRing R
    inst✝³ : IsDedekindDomain R
    inst✝² : Field K
    inst✝¹ : Algebra R K
    inst✝ : IsFractionRing R K
    a : DedekindDomain.FiniteAdeleRing R K
    S : Set (IsDedekindDomain.HeightOneSpectrum R) := setOf fun v => Not (Membersh …
    b : (v : IsDedekindDomain.HeightOneSpectrum R) → IsDedekindDomain.HeightOneSpe …
    hb : ∀ (v : IsDedekindDomain.HeightOneSpectrum R) (a : IsDedekindDomain.Height …
    h : ∀ (v : IsDedekindDomain.HeightOneSpectrum R) (a : IsDedekindDomain.HeightO …
    p : R := finprod fun v => finprod fun h => b v ((fun v => ↑a v) v)
    hp : Membership.mem (nonZeroDivisors R) p
    ⊢ ∀ (v : IsDedekindDomain.HeightOneSpectrum R), Membership.mem (IsDedekindDoma …
  -/
  intro v
  /-
    case h
    R : Type u_1
    K : Type u_2
    inst✝⁴ : CommRing R
    inst✝³ : IsDedekindDomain R
    inst✝² : Field K
    inst✝¹ : Algebra R K
    inst✝ : IsFractionRing R K
    a : DedekindDomain.FiniteAdeleRing R K
    S : Set (IsDedekindDomain.HeightOneSpectrum R) := setOf fun v => Not (Membersh …
    b : (v : IsDedekindDomain.HeightOneSpectrum R) → IsDedekindDomain.HeightOneSpe …
    hb : ∀ (v : IsDedekindDomain.HeightOneSpectrum R) (a : IsDedekindDomain.Height …
    h : ∀ (v : IsDedekindDomain.HeightOneSpectrum R) (a : IsDedekindDomain.HeightO …
    p : R := finprod fun v => finprod fun h => b v ((fun v => ↑a v) v)
    hp : Membership.mem (nonZeroDivisors R) p
    v : IsDedekindDomain.HeightOneSpectrum R
    ⊢ Membership.mem (IsDedekindDomain.HeightOneSpectrum.adicCompletionIntegers K  …
  -/
  by_cases hv : a v ∈ adicCompletionIntegers K v
    /-
      case pos
      R : Type u_1
      K : Type u_2
      inst✝⁴ : CommRing R
      inst✝³ : IsDedekindDomain R
      inst✝² : Field K
      inst✝¹ : Algebra R K
      inst✝ : IsFractionRing R K
      a : DedekindDomain.FiniteAdeleRing R K
      S : Set (IsDedekindDomain.HeightOneSpectrum R) := setOf fun v => Not (Membersh …
      b : (v : IsDedekindDomain.HeightOneSpectrum R) → IsDedekindDomain.HeightOneSpe …
      hb : ∀ (v : IsDedekindDomain.HeightOneSpectrum R) (a : IsDedekindDomain.Height …
      h : ∀ (v : IsDedekindDomain.HeightOneSpectrum R) (a : IsDedekindDomain.HeightO …
      p : R := finprod fun v => finprod fun h => b v ((fun v => ↑a v) v)
      hp : Membership.mem (nonZeroDivisors R) p
      v : IsDedekindDomain.HeightOneSpectrum R
      hv : Membership.mem (IsDedekindDomain.HeightOneSpectrum.adicCompletionIntegers …
      ⊢ Membership.mem (IsDedekindDomain.HeightOneSpectrum.adicCompletionIntegers K  …
    -/
  · exact mul_mem hv <| coe_mem_adicCompletionIntegers _ _
    /-
      🎉 no goals
    -/
    /-
      case neg
      R : Type u_1
      K : Type u_2
      inst✝⁴ : CommRing R
      inst✝³ : IsDedekindDomain R
      inst✝² : Field K
      inst✝¹ : Algebra R K
      inst✝ : IsFractionRing R K
      a : DedekindDomain.FiniteAdeleRing R K
      S : Set (IsDedekindDomain.HeightOneSpectrum R) := setOf fun v => Not (Membersh …
      b : (v : IsDedekindDomain.HeightOneSpectrum R) → IsDedekindDomain.HeightOneSpe …
      hb : ∀ (v : IsDedekindDomain.HeightOneSpectrum R) (a : IsDedekindDomain.Height …
      h : ∀ (v : IsDedekindDomain.HeightOneSpectrum R) (a : IsDedekindDomain.HeightO …
      p : R := finprod fun v => finprod fun h => b v ((fun v => ↑a v) v)
      hp : Membership.mem (nonZeroDivisors R) p
      v : IsDedekindDomain.HeightOneSpectrum R
      hv : Not (Membership.mem (IsDedekindDomain.HeightOneSpectrum.adicCompletionInt …
      ⊢ Membership.mem (IsDedekindDomain.HeightOneSpectrum.adicCompletionIntegers K  …
    -/
  · dsimp only
    have pprod : p = b v (a v) * ∏ᶠ w ∈ S \ {v}, b w (a w) := by
      rw [← finprod_mem_singleton (a := v) (f := fun v ↦ b v (a v)),
        finprod_mem_mul_diff (singleton_subset_iff.2 ‹v ∈ S›) a.2]
    /-
      case neg
      R : Type u_1
      K : Type u_2
      inst✝⁴ : CommRing R
      inst✝³ : IsDedekindDomain R
      inst✝² : Field K
      inst✝¹ : Algebra R K
      inst✝ : IsFractionRing R K
      a : DedekindDomain.FiniteAdeleRing R K
      S : Set (IsDedekindDomain.HeightOneSpectrum R) := setOf fun v => Not (Membersh …
      b : (v : IsDedekindDomain.HeightOneSpectrum R) → IsDedekindDomain.HeightOneSpe …
      hb : ∀ (v : IsDedekindDomain.HeightOneSpectrum R) (a : IsDedekindDomain.Height …
      h : ∀ (v : IsDedekindDomain.HeightOneSpectrum R) (a : IsDedekindDomain.HeightO …
      p : R := finprod fun v => finprod fun h => b v ((fun v => ↑a v) v)
      hp : Membership.mem (nonZeroDivisors R) p
      v : IsDedekindDomain.HeightOneSpectrum R
      hv : Not (Membership.mem (IsDedekindDomain.HeightOneSpectrum.adicCompletionInt …
      pprod : Eq p (HMul.hMul (b v ((fun v => ↑a v) v)) (finprod fun w => finprod fu …
      ⊢ Membership.mem (IsDedekindDomain.HeightOneSpectrum.adicCompletionIntegers K  …
    -/
    rw [pprod]
    /-
      case neg
      R : Type u_1
      K : Type u_2
      inst✝⁴ : CommRing R
      inst✝³ : IsDedekindDomain R
      inst✝² : Field K
      inst✝¹ : Algebra R K
      inst✝ : IsFractionRing R K
      a : DedekindDomain.FiniteAdeleRing R K
      S : Set (IsDedekindDomain.HeightOneSpectrum R) := setOf fun v => Not (Membersh …
      b : (v : IsDedekindDomain.HeightOneSpectrum R) → IsDedekindDomain.HeightOneSpe …
      hb : ∀ (v : IsDedekindDomain.HeightOneSpectrum R) (a : IsDedekindDomain.Height …
      h : ∀ (v : IsDedekindDomain.HeightOneSpectrum R) (a : IsDedekindDomain.HeightO …
      p : R := finprod fun v => finprod fun h => b v ((fun v => ↑a v) v)
      hp : Membership.mem (nonZeroDivisors R) p
      v : IsDedekindDomain.HeightOneSpectrum R
      hv : Not (Membership.mem (IsDedekindDomain.HeightOneSpectrum.adicCompletionInt …
      pprod : Eq p (HMul.hMul (b v ((fun v => ↑a v) v)) (finprod fun w => finprod fu …
      ⊢ Membership.mem (IsDedekindDomain.HeightOneSpectrum.adicCompletionIntegers K  …
    -/
    push_cast
    /-
      case neg
      R : Type u_1
      K : Type u_2
      inst✝⁴ : CommRing R
      inst✝³ : IsDedekindDomain R
      inst✝² : Field K
      inst✝¹ : Algebra R K
      inst✝ : IsFractionRing R K
      a : DedekindDomain.FiniteAdeleRing R K
      S : Set (IsDedekindDomain.HeightOneSpectrum R) := setOf fun v => Not (Membersh …
      b : (v : IsDedekindDomain.HeightOneSpectrum R) → IsDedekindDomain.HeightOneSpe …
      hb : ∀ (v : IsDedekindDomain.HeightOneSpectrum R) (a : IsDedekindDomain.Height …
      h : ∀ (v : IsDedekindDomain.HeightOneSpectrum R) (a : IsDedekindDomain.HeightO …
      p : R := finprod fun v => finprod fun h => b v ((fun v => ↑a v) v)
      hp : Membership.mem (nonZeroDivisors R) p
      v : IsDedekindDomain.HeightOneSpectrum R
      hv : Not (Membership.mem (IsDedekindDomain.HeightOneSpectrum.adicCompletionInt …
      pprod : Eq p (HMul.hMul (b v ((fun v => ↑a v) v)) (finprod fun w => finprod fu …
      ⊢ Membership.mem (IsDedekindDomain.HeightOneSpectrum.adicCompletionIntegers K  …
    -/
    rw [← mul_assoc]
    /-
      case neg
      R : Type u_1
      K : Type u_2
      inst✝⁴ : CommRing R
      inst✝³ : IsDedekindDomain R
      inst✝² : Field K
      inst✝¹ : Algebra R K
      inst✝ : IsFractionRing R K
      a : DedekindDomain.FiniteAdeleRing R K
      S : Set (IsDedekindDomain.HeightOneSpectrum R) := setOf fun v => Not (Membersh …
      b : (v : IsDedekindDomain.HeightOneSpectrum R) → IsDedekindDomain.HeightOneSpe …
      hb : ∀ (v : IsDedekindDomain.HeightOneSpectrum R) (a : IsDedekindDomain.Height …
      h : ∀ (v : IsDedekindDomain.HeightOneSpectrum R) (a : IsDedekindDomain.HeightO …
      p : R := finprod fun v => finprod fun h => b v ((fun v => ↑a v) v)
      hp : Membership.mem (nonZeroDivisors R) p
      v : IsDedekindDomain.HeightOneSpectrum R
      hv : Not (Membership.mem (IsDedekindDomain.HeightOneSpectrum.adicCompletionInt …
      pprod : Eq p (HMul.hMul (b v ((fun v => ↑a v) v)) (finprod fun w => finprod fu …
      ⊢ Membership.mem (IsDedekindDomain.HeightOneSpectrum.adicCompletionIntegers K  …
    -/
    exact mul_mem (h v (a v)) <| coe_mem_adicCompletionIntegers _ _
    /-
      🎉 no goals
    -/


theorem submodulesRingBasis : SubmodulesRingBasis
    (fun (r : R⁰) ↦ Submodule.span (R_hat R K) {((r : R) : FiniteAdeleRing R K)}) where
  inter i j := ⟨i * j, by
    /-
      R : Type u_1
      K : Type u_2
      inst✝⁴ : CommRing R
      inst✝³ : IsDedekindDomain R
      inst✝² : Field K
      inst✝¹ : Algebra R K
      inst✝ : IsFractionRing R K
      i j : Subtype fun x => Membership.mem (nonZeroDivisors R) x
      ⊢ LE.le (Submodule.span (DedekindDomain.FiniteIntegralAdeles R K) (Singleton.s …
    -/
    push_cast
    /-
      R : Type u_1
      K : Type u_2
      inst✝⁴ : CommRing R
      inst✝³ : IsDedekindDomain R
      inst✝² : Field K
      inst✝¹ : Algebra R K
      inst✝ : IsFractionRing R K
      i j : Subtype fun x => Membership.mem (nonZeroDivisors R) x
      ⊢ LE.le (Submodule.span (DedekindDomain.FiniteIntegralAdeles R K) (Singleton.s …
    -/
    simp only [le_inf_iff, Submodule.span_singleton_le_iff_mem, Submodule.mem_span_singleton]
    /-
      R : Type u_1
      K : Type u_2
      inst✝⁴ : CommRing R
      inst✝³ : IsDedekindDomain R
      inst✝² : Field K
      inst✝¹ : Algebra R K
      inst✝ : IsFractionRing R K
      i j : Subtype fun x => Membership.mem (nonZeroDivisors R) x
      ⊢ And (Exists fun a => Eq (HSMul.hSMul a ↑↑i) (HMul.hMul ↑↑i ↑↑j)) (Exists fun …
    -/
    exact ⟨⟨((j : R) : R_hat R K), by rw [mul_comm]; rfl⟩, ⟨((i : R) : R_hat R K), rfl⟩⟩⟩
    /-
      🎉 no goals
    -/
  leftMul a r := by
    /-
      R : Type u_1
      K : Type u_2
      inst✝⁴ : CommRing R
      inst✝³ : IsDedekindDomain R
      inst✝² : Field K
      inst✝¹ : Algebra R K
      inst✝ : IsFractionRing R K
      a : DedekindDomain.FiniteAdeleRing R K
      r : Subtype fun x => Membership.mem (nonZeroDivisors R) x
      ⊢ Exists fun j => LE.le (HSMul.hSMul a (Submodule.span (DedekindDomain.FiniteI …
    -/
    rcases mul_nonZeroDivisor_mem_finiteIntegralAdeles a with ⟨b, c, h⟩
    /-
      case intro.intro
      R : Type u_1
      K : Type u_2
      inst✝⁴ : CommRing R
      inst✝³ : IsDedekindDomain R
      inst✝² : Field K
      inst✝¹ : Algebra R K
      inst✝ : IsFractionRing R K
      a : DedekindDomain.FiniteAdeleRing R K
      r b : Subtype fun x => Membership.mem (nonZeroDivisors R) x
      c : DedekindDomain.FiniteIntegralAdeles R K
      h : Eq (HMul.hMul a ↑↑b) ↑c
      ⊢ Exists fun j => LE.le (HSMul.hSMul a (Submodule.span (DedekindDomain.FiniteI …
    -/
    use r * b
    /-
      case h
      R : Type u_1
      K : Type u_2
      inst✝⁴ : CommRing R
      inst✝³ : IsDedekindDomain R
      inst✝² : Field K
      inst✝¹ : Algebra R K
      inst✝ : IsFractionRing R K
      a : DedekindDomain.FiniteAdeleRing R K
      r b : Subtype fun x => Membership.mem (nonZeroDivisors R) x
      c : DedekindDomain.FiniteIntegralAdeles R K
      h : Eq (HMul.hMul a ↑↑b) ↑c
      ⊢ LE.le (HSMul.hSMul a (Submodule.span (DedekindDomain.FiniteIntegralAdeles R  …
    -/
    rintro x ⟨m, hm, rfl⟩
    /-
      case h.intro.intro
      R : Type u_1
      K : Type u_2
      inst✝⁴ : CommRing R
      inst✝³ : IsDedekindDomain R
      inst✝² : Field K
      inst✝¹ : Algebra R K
      inst✝ : IsFractionRing R K
      a : DedekindDomain.FiniteAdeleRing R K
      r b : Subtype fun x => Membership.mem (nonZeroDivisors R) x
      c : DedekindDomain.FiniteIntegralAdeles R K
      h : Eq (HMul.hMul a ↑↑b) ↑c
      m : DedekindDomain.FiniteAdeleRing R K
      hm : Membership.mem (↑(Submodule.span (DedekindDomain.FiniteIntegralAdeles R K …
      ⊢ Membership.mem (Submodule.span (DedekindDomain.FiniteIntegralAdeles R K) (Si …
    -/
    simp only [Submonoid.coe_mul, SetLike.mem_coe] at hm
    /-
      case h.intro.intro
      R : Type u_1
      K : Type u_2
      inst✝⁴ : CommRing R
      inst✝³ : IsDedekindDomain R
      inst✝² : Field K
      inst✝¹ : Algebra R K
      inst✝ : IsFractionRing R K
      a : DedekindDomain.FiniteAdeleRing R K
      r b : Subtype fun x => Membership.mem (nonZeroDivisors R) x
      c : DedekindDomain.FiniteIntegralAdeles R K
      h : Eq (HMul.hMul a ↑↑b) ↑c
      m : DedekindDomain.FiniteAdeleRing R K
      hm : Membership.mem (Submodule.span (DedekindDomain.FiniteIntegralAdeles R K)  …
      ⊢ Membership.mem (Submodule.span (DedekindDomain.FiniteIntegralAdeles R K) (Si …
    -/
    rw [Submodule.mem_span_singleton] at hm ⊢
    /-
      case h.intro.intro
      R : Type u_1
      K : Type u_2
      inst✝⁴ : CommRing R
      inst✝³ : IsDedekindDomain R
      inst✝² : Field K
      inst✝¹ : Algebra R K
      inst✝ : IsFractionRing R K
      a : DedekindDomain.FiniteAdeleRing R K
      r b : Subtype fun x => Membership.mem (nonZeroDivisors R) x
      c : DedekindDomain.FiniteIntegralAdeles R K
      h : Eq (HMul.hMul a ↑↑b) ↑c
      m : DedekindDomain.FiniteAdeleRing R K
      hm : Exists fun a => Eq (HSMul.hSMul a ↑(HMul.hMul ↑r ↑b)) m
      ⊢ Exists fun a_1 => Eq (HSMul.hSMul a_1 ↑↑r) ((DistribMulAction.toLinearMap (D …
    -/
    rcases hm with ⟨n, rfl⟩
    /-
      case h.intro.intro.intro
      R : Type u_1
      K : Type u_2
      inst✝⁴ : CommRing R
      inst✝³ : IsDedekindDomain R
      inst✝² : Field K
      inst✝¹ : Algebra R K
      inst✝ : IsFractionRing R K
      a : DedekindDomain.FiniteAdeleRing R K
      r b : Subtype fun x => Membership.mem (nonZeroDivisors R) x
      c : DedekindDomain.FiniteIntegralAdeles R K
      h : Eq (HMul.hMul a ↑↑b) ↑c
      n : DedekindDomain.FiniteIntegralAdeles R K
      ⊢ Exists fun a_1 => Eq (HSMul.hSMul a_1 ↑↑r) ((DistribMulAction.toLinearMap (D …
    -/
    simp only [LinearMapClass.map_smul, DistribMulAction.toLinearMap_apply, smul_eq_mul]
    /-
      case h.intro.intro.intro
      R : Type u_1
      K : Type u_2
      inst✝⁴ : CommRing R
      inst✝³ : IsDedekindDomain R
      inst✝² : Field K
      inst✝¹ : Algebra R K
      inst✝ : IsFractionRing R K
      a : DedekindDomain.FiniteAdeleRing R K
      r b : Subtype fun x => Membership.mem (nonZeroDivisors R) x
      c : DedekindDomain.FiniteIntegralAdeles R K
      h : Eq (HMul.hMul a ↑↑b) ↑c
      n : DedekindDomain.FiniteIntegralAdeles R K
      ⊢ Exists fun a_1 => Eq (HSMul.hSMul a_1 ↑↑r) (HSMul.hSMul n (HMul.hMul a ↑(HMu …
    -/
    use n * c
    /-
      case h
      R : Type u_1
      K : Type u_2
      inst✝⁴ : CommRing R
      inst✝³ : IsDedekindDomain R
      inst✝² : Field K
      inst✝¹ : Algebra R K
      inst✝ : IsFractionRing R K
      a : DedekindDomain.FiniteAdeleRing R K
      r b : Subtype fun x => Membership.mem (nonZeroDivisors R) x
      c : DedekindDomain.FiniteIntegralAdeles R K
      h : Eq (HMul.hMul a ↑↑b) ↑c
      n : DedekindDomain.FiniteIntegralAdeles R K
      ⊢ Eq (HSMul.hSMul (HMul.hMul n c) ↑↑r) (HSMul.hSMul n (HMul.hMul a ↑(HMul.hMul …
    -/
    push_cast
    rw [mul_left_comm, h, mul_comm _ (c : FiniteAdeleRing R K),
      Algebra.smul_def', Algebra.smul_def', ← mul_assoc]
    /-
      case h
      R : Type u_1
      K : Type u_2
      inst✝⁴ : CommRing R
      inst✝³ : IsDedekindDomain R
      inst✝² : Field K
      inst✝¹ : Algebra R K
      inst✝ : IsFractionRing R K
      a : DedekindDomain.FiniteAdeleRing R K
      r b : Subtype fun x => Membership.mem (nonZeroDivisors R) x
      c : DedekindDomain.FiniteIntegralAdeles R K
      h : Eq (HMul.hMul a ↑↑b) ↑c
      n : DedekindDomain.FiniteIntegralAdeles R K
      ⊢ Eq (HMul.hMul (Algebra.toRingHom (HMul.hMul n c)) ↑↑r) (HMul.hMul (HMul.hMul …
    -/
    rfl
    /-
      🎉 no goals
    -/
  mul r := ⟨r, by
    /-
      R : Type u_1
      K : Type u_2
      inst✝⁴ : CommRing R
      inst✝³ : IsDedekindDomain R
      inst✝² : Field K
      inst✝¹ : Algebra R K
      inst✝ : IsFractionRing R K
      r : Subtype fun x => Membership.mem (nonZeroDivisors R) x
      ⊢ HasSubset.Subset (HMul.hMul ↑(Submodule.span (DedekindDomain.FiniteIntegralA …
    -/
    intro x hx
    /-
      R : Type u_1
      K : Type u_2
      inst✝⁴ : CommRing R
      inst✝³ : IsDedekindDomain R
      inst✝² : Field K
      inst✝¹ : Algebra R K
      inst✝ : IsFractionRing R K
      r : Subtype fun x => Membership.mem (nonZeroDivisors R) x
      x : DedekindDomain.FiniteAdeleRing R K
      hx : Membership.mem (HMul.hMul ↑(Submodule.span (DedekindDomain.FiniteIntegral …
      ⊢ Membership.mem (↑(Submodule.span (DedekindDomain.FiniteIntegralAdeles R K) ( …
    -/
    rw [mem_mul] at hx
    /-
      R : Type u_1
      K : Type u_2
      inst✝⁴ : CommRing R
      inst✝³ : IsDedekindDomain R
      inst✝² : Field K
      inst✝¹ : Algebra R K
      inst✝ : IsFractionRing R K
      r : Subtype fun x => Membership.mem (nonZeroDivisors R) x
      x : DedekindDomain.FiniteAdeleRing R K
      hx : Exists fun x_1 => And (Membership.mem (↑(Submodule.span (DedekindDomain.F …
      ⊢ Membership.mem (↑(Submodule.span (DedekindDomain.FiniteIntegralAdeles R K) ( …
    -/
    rcases hx with ⟨a, ha, b, hb, rfl⟩
    /-
      case intro.intro.intro.intro
      R : Type u_1
      K : Type u_2
      inst✝⁴ : CommRing R
      inst✝³ : IsDedekindDomain R
      inst✝² : Field K
      inst✝¹ : Algebra R K
      inst✝ : IsFractionRing R K
      r : Subtype fun x => Membership.mem (nonZeroDivisors R) x
      a : DedekindDomain.FiniteAdeleRing R K
      ha : Membership.mem (↑(Submodule.span (DedekindDomain.FiniteIntegralAdeles R K …
      b : DedekindDomain.FiniteAdeleRing R K
      hb : Membership.mem (↑(Submodule.span (DedekindDomain.FiniteIntegralAdeles R K …
      ⊢ Membership.mem (↑(Submodule.span (DedekindDomain.FiniteIntegralAdeles R K) ( …
    -/
    simp only [SetLike.mem_coe, Submodule.mem_span_singleton] at ha hb ⊢
    /-
      case intro.intro.intro.intro
      R : Type u_1
      K : Type u_2
      inst✝⁴ : CommRing R
      inst✝³ : IsDedekindDomain R
      inst✝² : Field K
      inst✝¹ : Algebra R K
      inst✝ : IsFractionRing R K
      r : Subtype fun x => Membership.mem (nonZeroDivisors R) x
      a b : DedekindDomain.FiniteAdeleRing R K
      ha : Exists fun a_1 => Eq (HSMul.hSMul a_1 ↑↑r) a
      hb : Exists fun a => Eq (HSMul.hSMul a ↑↑r) b
      ⊢ Exists fun a_1 => Eq (HSMul.hSMul a_1 ↑↑r) (HMul.hMul a b)
    -/
    rcases ha with ⟨m, rfl⟩
    /-
      case intro.intro.intro.intro.intro
      R : Type u_1
      K : Type u_2
      inst✝⁴ : CommRing R
      inst✝³ : IsDedekindDomain R
      inst✝² : Field K
      inst✝¹ : Algebra R K
      inst✝ : IsFractionRing R K
      r : Subtype fun x => Membership.mem (nonZeroDivisors R) x
      b : DedekindDomain.FiniteAdeleRing R K
      hb : Exists fun a => Eq (HSMul.hSMul a ↑↑r) b
      m : DedekindDomain.FiniteIntegralAdeles R K
      ⊢ Exists fun a => Eq (HSMul.hSMul a ↑↑r) (HMul.hMul (HSMul.hSMul m ↑↑r) b)
    -/
    rcases hb with ⟨n, rfl⟩
    /-
      case intro.intro.intro.intro.intro.intro
      R : Type u_1
      K : Type u_2
      inst✝⁴ : CommRing R
      inst✝³ : IsDedekindDomain R
      inst✝² : Field K
      inst✝¹ : Algebra R K
      inst✝ : IsFractionRing R K
      r : Subtype fun x => Membership.mem (nonZeroDivisors R) x
      m n : DedekindDomain.FiniteIntegralAdeles R K
      ⊢ Exists fun a => Eq (HSMul.hSMul a ↑↑r) (HMul.hMul (HSMul.hSMul m ↑↑r) (HSMul …
    -/
    use m * n * (r : R)
    /-
      case h
      R : Type u_1
      K : Type u_2
      inst✝⁴ : CommRing R
      inst✝³ : IsDedekindDomain R
      inst✝² : Field K
      inst✝¹ : Algebra R K
      inst✝ : IsFractionRing R K
      r : Subtype fun x => Membership.mem (nonZeroDivisors R) x
      m n : DedekindDomain.FiniteIntegralAdeles R K
      ⊢ Eq (HSMul.hSMul (HMul.hMul (HMul.hMul m n) ↑↑r) ↑↑r) (HMul.hMul (HSMul.hSMul …
    -/
    simp only [Algebra.smul_def', map_mul]
    /-
      case h
      R : Type u_1
      K : Type u_2
      inst✝⁴ : CommRing R
      inst✝³ : IsDedekindDomain R
      inst✝² : Field K
      inst✝¹ : Algebra R K
      inst✝ : IsFractionRing R K
      r : Subtype fun x => Membership.mem (nonZeroDivisors R) x
      m n : DedekindDomain.FiniteIntegralAdeles R K
      ⊢ Eq (HMul.hMul (HMul.hMul (HMul.hMul (Algebra.toRingHom m) (Algebra.toRingHom …
    -/
    rw [mul_mul_mul_comm, mul_assoc]
    /-
      case h
      R : Type u_1
      K : Type u_2
      inst✝⁴ : CommRing R
      inst✝³ : IsDedekindDomain R
      inst✝² : Field K
      inst✝¹ : Algebra R K
      inst✝ : IsFractionRing R K
      r : Subtype fun x => Membership.mem (nonZeroDivisors R) x
      m n : DedekindDomain.FiniteIntegralAdeles R K
      ⊢ Eq (HMul.hMul (HMul.hMul (Algebra.toRingHom m) (Algebra.toRingHom n)) (HMul. …
    -/
    rfl⟩
    /-
      🎉 no goals
    -/


instance : TopologicalSpace (FiniteAdeleRing R K) :=
  SubmodulesRingBasis.topology (submodulesRingBasis R K)

-- the point of `submodulesRingBasis` above: this now works

