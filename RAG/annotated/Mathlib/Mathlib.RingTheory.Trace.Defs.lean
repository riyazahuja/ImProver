/-- The trace of an element `s` of an `R`-algebra is the trace of `(s * ·)`,
as an `R`-linear map. -/
@[stacks 0BIF "Trace"]
noncomputable def trace : S →ₗ[R] R :=
  (LinearMap.trace R S).comp (lmul R S).toLinearMap


theorem trace_apply (x) : trace R S x = LinearMap.trace R S (lmul R S x) :=
  rfl


theorem trace_eq_zero_of_not_exists_basis (h : ¬∃ s : Finset S, Nonempty (Basis s R S)) :
                        /-
                          R : Type u_1
                          S : Type u_2
                          inst✝² : CommRing R
                          inst✝¹ : CommRing S
                          inst✝ : Algebra R S
                          h : Not (Exists fun s => Nonempty (Basis (Subtype fun x => Membership.mem s x) …
                          ⊢ Eq (Algebra.trace R S) 0
                        -/
    trace R S = 0 := by ext s; simp [trace_apply, LinearMap.trace, h]
                               /-
                                 🎉 no goals
                               -/


theorem trace_eq_matrix_trace [DecidableEq ι] (b : Basis ι R S) (s : S) :
    trace R S s = Matrix.trace (Algebra.leftMulMatrix b s) := by
  /-
    R : Type u_1
    S : Type u_2
    inst✝⁴ : CommRing R
    inst✝³ : CommRing S
    inst✝² : Algebra R S
    ι : Type w
    inst✝¹ : Fintype ι
    inst✝ : DecidableEq ι
    b : Basis ι R S
    s : S
    ⊢ Eq ((Algebra.trace R S) s) ((Algebra.leftMulMatrix b) s).trace
  -/
  rw [trace_apply, LinearMap.trace_eq_matrix_trace _ b, ← toMatrix_lmul_eq]; rfl
                                                                             /-
                                                                               🎉 no goals
                                                                             -/


/-- If `x` is in the base field `K`, then the trace is `[L : K] * x`. -/
theorem trace_algebraMap_of_basis (b : Basis ι R S) (x : R) :
    trace R S (algebraMap R S x) = Fintype.card ι • x := by
  /-
    R : Type u_1
    S : Type u_2
    inst✝³ : CommRing R
    inst✝² : CommRing S
    inst✝¹ : Algebra R S
    ι : Type w
    inst✝ : Fintype ι
    b : Basis ι R S
    x : R
    ⊢ Eq ((Algebra.trace R S) ((algebraMap R S) x)) (HSMul.hSMul (Fintype.card ι) x)
  -/
  haveI := Classical.decEq ι
  /-
    R : Type u_1
    S : Type u_2
    inst✝³ : CommRing R
    inst✝² : CommRing S
    inst✝¹ : Algebra R S
    ι : Type w
    inst✝ : Fintype ι
    b : Basis ι R S
    x : R
    this : DecidableEq ι
    ⊢ Eq ((Algebra.trace R S) ((algebraMap R S) x)) (HSMul.hSMul (Fintype.card ι) x)
  -/
  rw [trace_apply, LinearMap.trace_eq_matrix_trace R b, Matrix.trace]
  /-
    R : Type u_1
    S : Type u_2
    inst✝³ : CommRing R
    inst✝² : CommRing S
    inst✝¹ : Algebra R S
    ι : Type w
    inst✝ : Fintype ι
    b : Basis ι R S
    x : R
    this : DecidableEq ι
    ⊢ Eq (Finset.univ.sum fun i => ((LinearMap.toMatrix b b) ((Algebra.lmul R S) ( …
  -/
  convert Finset.sum_const x
  /-
    case h.e'_2.a
    R : Type u_1
    S : Type u_2
    inst✝³ : CommRing R
    inst✝² : CommRing S
    inst✝¹ : Algebra R S
    ι : Type w
    inst✝ : Fintype ι
    b : Basis ι R S
    x : R
    this : DecidableEq ι
    x✝ : ι
    a✝ : Membership.mem Finset.univ x✝
    ⊢ Eq (((LinearMap.toMatrix b b) ((Algebra.lmul R S) ((algebraMap R S) x))).dia …
  -/
  simp [-coe_lmul_eq_mul]
  /-
    🎉 no goals
  -/


/-- If `x` is in the base field `K`, then the trace is `[L : K] * x`.

(If `L` is not finite-dimensional over `K`, then `trace` and `finrank` return `0`.)
-/
@[simp]
theorem trace_algebraMap [StrongRankCondition R] [Module.Free R S] (x : R) :
    trace R S (algebraMap R S x) = finrank R S • x := by
  /-
    R : Type u_1
    S : Type u_2
    inst✝⁴ : CommRing R
    inst✝³ : CommRing S
    inst✝² : Algebra R S
    inst✝¹ : StrongRankCondition R
    inst✝ : Module.Free R S
    x : R
    ⊢ Eq ((Algebra.trace R S) ((algebraMap R S) x)) (HSMul.hSMul (Module.finrank R …
  -/
  by_cases H : ∃ s : Finset S, Nonempty (Basis s R S)
    /-
      case pos
      R : Type u_1
      S : Type u_2
      inst✝⁴ : CommRing R
      inst✝³ : CommRing S
      inst✝² : Algebra R S
      inst✝¹ : StrongRankCondition R
      inst✝ : Module.Free R S
      x : R
      H : Exists fun s => Nonempty (Basis (Subtype fun x => Membership.mem s x) R S)
      ⊢ Eq ((Algebra.trace R S) ((algebraMap R S) x)) (HSMul.hSMul (Module.finrank R …
    -/
  · rw [trace_algebraMap_of_basis H.choose_spec.some, finrank_eq_card_basis H.choose_spec.some]
    /-
      🎉 no goals
    -/
    /-
      case neg
      R : Type u_1
      S : Type u_2
      inst✝⁴ : CommRing R
      inst✝³ : CommRing S
      inst✝² : Algebra R S
      inst✝¹ : StrongRankCondition R
      inst✝ : Module.Free R S
      x : R
      H : Not (Exists fun s => Nonempty (Basis (Subtype fun x => Membership.mem s x) …
      ⊢ Eq ((Algebra.trace R S) ((algebraMap R S) x)) (HSMul.hSMul (Module.finrank R …
    -/
  · simp [trace_eq_zero_of_not_exists_basis R H, finrank_eq_zero_of_not_exists_basis_finset H]
    /-
      🎉 no goals
    -/


theorem trace_trace_of_basis [Algebra S T] [IsScalarTower R S T] {ι κ : Type*} [Finite ι]
    [Finite κ] (b : Basis ι R S) (c : Basis κ S T) (x : T) :
    trace R S (trace S T x) = trace R T x := by
  /-
    R : Type u_1
    S : Type u_2
    T : Type u_3
    inst✝⁸ : CommRing R
    inst✝⁷ : CommRing S
    inst✝⁶ : CommRing T
    inst✝⁵ : Algebra R S
    inst✝⁴ : Algebra R T
    inst✝³ : Algebra S T
    inst✝² : IsScalarTower R S T
    ι : Type u_4
    κ : Type u_5
    inst✝¹ : Finite ι
    inst✝ : Finite κ
    b : Basis ι R S
    c : Basis κ S T
    x : T
    ⊢ Eq ((Algebra.trace R S) ((Algebra.trace S T) x)) ((Algebra.trace R T) x)
  -/
  haveI := Classical.decEq ι
  /-
    R : Type u_1
    S : Type u_2
    T : Type u_3
    inst✝⁸ : CommRing R
    inst✝⁷ : CommRing S
    inst✝⁶ : CommRing T
    inst✝⁵ : Algebra R S
    inst✝⁴ : Algebra R T
    inst✝³ : Algebra S T
    inst✝² : IsScalarTower R S T
    ι : Type u_4
    κ : Type u_5
    inst✝¹ : Finite ι
    inst✝ : Finite κ
    b : Basis ι R S
    c : Basis κ S T
    x : T
    this : DecidableEq ι
    ⊢ Eq ((Algebra.trace R S) ((Algebra.trace S T) x)) ((Algebra.trace R T) x)
  -/
  haveI := Classical.decEq κ
  /-
    R : Type u_1
    S : Type u_2
    T : Type u_3
    inst✝⁸ : CommRing R
    inst✝⁷ : CommRing S
    inst✝⁶ : CommRing T
    inst✝⁵ : Algebra R S
    inst✝⁴ : Algebra R T
    inst✝³ : Algebra S T
    inst✝² : IsScalarTower R S T
    ι : Type u_4
    κ : Type u_5
    inst✝¹ : Finite ι
    inst✝ : Finite κ
    b : Basis ι R S
    c : Basis κ S T
    x : T
    this✝ : DecidableEq ι
    this : DecidableEq κ
    ⊢ Eq ((Algebra.trace R S) ((Algebra.trace S T) x)) ((Algebra.trace R T) x)
  -/
  cases nonempty_fintype ι
  /-
    case intro
    R : Type u_1
    S : Type u_2
    T : Type u_3
    inst✝⁸ : CommRing R
    inst✝⁷ : CommRing S
    inst✝⁶ : CommRing T
    inst✝⁵ : Algebra R S
    inst✝⁴ : Algebra R T
    inst✝³ : Algebra S T
    inst✝² : IsScalarTower R S T
    ι : Type u_4
    κ : Type u_5
    inst✝¹ : Finite ι
    inst✝ : Finite κ
    b : Basis ι R S
    c : Basis κ S T
    x : T
    this✝ : DecidableEq ι
    this : DecidableEq κ
    val✝ : Fintype ι
    ⊢ Eq ((Algebra.trace R S) ((Algebra.trace S T) x)) ((Algebra.trace R T) x)
  -/
  cases nonempty_fintype κ
  rw [trace_eq_matrix_trace (b.smulTower c), trace_eq_matrix_trace b, trace_eq_matrix_trace c,
    Matrix.trace, Matrix.trace, Matrix.trace, ← Finset.univ_product_univ, Finset.sum_product]
  /-
    case intro.intro
    R : Type u_1
    S : Type u_2
    T : Type u_3
    inst✝⁸ : CommRing R
    inst✝⁷ : CommRing S
    inst✝⁶ : CommRing T
    inst✝⁵ : Algebra R S
    inst✝⁴ : Algebra R T
    inst✝³ : Algebra S T
    inst✝² : IsScalarTower R S T
    ι : Type u_4
    κ : Type u_5
    inst✝¹ : Finite ι
    inst✝ : Finite κ
    b : Basis ι R S
    c : Basis κ S T
    x : T
    this✝ : DecidableEq ι
    this : DecidableEq κ
    val✝¹ : Fintype ι
    val✝ : Fintype κ
    ⊢ Eq (Finset.univ.sum fun i => ((Algebra.leftMulMatrix b) (Finset.univ.sum fun …
  -/
  refine Finset.sum_congr rfl fun i _ ↦ ?_
  simp only [map_sum, smulTower_leftMulMatrix, Finset.sum_apply, Matrix.diag,
    Finset.sum_apply i (Finset.univ : Finset κ) fun y => leftMulMatrix b (leftMulMatrix c x y y)]


theorem trace_comp_trace_of_basis [Algebra S T] [IsScalarTower R S T] {ι κ : Type*} [Finite ι]
    [Finite κ] (b : Basis ι R S) (c : Basis κ S T) :
    (trace R S).comp ((trace S T).restrictScalars R) = trace R T := by
  /-
    R : Type u_1
    S : Type u_2
    T : Type u_3
    inst✝⁸ : CommRing R
    inst✝⁷ : CommRing S
    inst✝⁶ : CommRing T
    inst✝⁵ : Algebra R S
    inst✝⁴ : Algebra R T
    inst✝³ : Algebra S T
    inst✝² : IsScalarTower R S T
    ι : Type u_4
    κ : Type u_5
    inst✝¹ : Finite ι
    inst✝ : Finite κ
    b : Basis ι R S
    c : Basis κ S T
    ⊢ Eq ((Algebra.trace R S).comp (↑R (Algebra.trace S T))) (Algebra.trace R T)
  -/
  ext
  /-
    case h
    R : Type u_1
    S : Type u_2
    T : Type u_3
    inst✝⁸ : CommRing R
    inst✝⁷ : CommRing S
    inst✝⁶ : CommRing T
    inst✝⁵ : Algebra R S
    inst✝⁴ : Algebra R T
    inst✝³ : Algebra S T
    inst✝² : IsScalarTower R S T
    ι : Type u_4
    κ : Type u_5
    inst✝¹ : Finite ι
    inst✝ : Finite κ
    b : Basis ι R S
    c : Basis κ S T
    x✝ : T
    ⊢ Eq (((Algebra.trace R S).comp (↑R (Algebra.trace S T))) x✝) ((Algebra.trace  …
  -/
  rw [LinearMap.comp_apply, LinearMap.restrictScalars_apply, trace_trace_of_basis b c]
  /-
    🎉 no goals
  -/


@[simp]
theorem trace_trace [Algebra S T] [IsScalarTower R S T]
    [Module.Free R S] [Module.Finite R S] [Module.Free S T] [Module.Finite S T] (x : T) :
    trace R S (trace S T x) = trace R T x :=
  trace_trace_of_basis (Module.Free.chooseBasis R S) (Module.Free.chooseBasis S T) x


/-- Let `T / S / R` be a tower of finite extensions of fields. Then
$\text{Trace}_{T/R} = \text{Trace}_{S/R} \circ \text{Trace}_{T/S}$.-/
@[simp, stacks 0BIJ "Trace"]
theorem trace_comp_trace [Algebra S T] [IsScalarTower R S T]
    [Module.Free R S] [Module.Finite R S] [Module.Free S T] [Module.Finite S T] :
    (trace R S).comp ((trace S T).restrictScalars R) = trace R T :=
  LinearMap.ext trace_trace


@[simp]
theorem trace_prod_apply [Module.Free R S] [Module.Free R T] [Module.Finite R S] [Module.Finite R T]
    (x : S × T) : trace R (S × T) x = trace R S x.fst + trace R T x.snd := by
  /-
    R : Type u_1
    S : Type u_2
    T : Type u_3
    inst✝⁸ : CommRing R
    inst✝⁷ : CommRing S
    inst✝⁶ : CommRing T
    inst✝⁵ : Algebra R S
    inst✝⁴ : Algebra R T
    inst✝³ : Module.Free R S
    inst✝² : Module.Free R T
    inst✝¹ : Module.Finite R S
    inst✝ : Module.Finite R T
    x : Prod S T
    ⊢ Eq ((Algebra.trace R (Prod S T)) x) (HAdd.hAdd ((Algebra.trace R S) x.1) ((A …
  -/
  nontriviality R
  /-
    R : Type u_1
    S : Type u_2
    T : Type u_3
    inst✝⁸ : CommRing R
    inst✝⁷ : CommRing S
    inst✝⁶ : CommRing T
    inst✝⁵ : Algebra R S
    inst✝⁴ : Algebra R T
    inst✝³ : Module.Free R S
    inst✝² : Module.Free R T
    inst✝¹ : Module.Finite R S
    inst✝ : Module.Finite R T
    x : Prod S T
    a✝ : Nontrivial R
    ⊢ Eq ((Algebra.trace R (Prod S T)) x) (HAdd.hAdd ((Algebra.trace R S) x.1) ((A …
  -/
  let f := (lmul R S).toLinearMap.prodMap (lmul R T).toLinearMap
  have : (lmul R (S × T)).toLinearMap = (prodMapLinear R S T S T R).comp f :=
    LinearMap.ext₂ Prod.mul_def
  /-
    R : Type u_1
    S : Type u_2
    T : Type u_3
    inst✝⁸ : CommRing R
    inst✝⁷ : CommRing S
    inst✝⁶ : CommRing T
    inst✝⁵ : Algebra R S
    inst✝⁴ : Algebra R T
    inst✝³ : Module.Free R S
    inst✝² : Module.Free R T
    inst✝¹ : Module.Finite R S
    inst✝ : Module.Finite R T
    x : Prod S T
    a✝ : Nontrivial R
    f : LinearMap (RingHom.id R) (Prod S T) (Prod (Module.End R S) (Module.End R T …
    this : Eq (Algebra.lmul R (Prod S T)).toLinearMap ((LinearMap.prodMapLinear R  …
    ⊢ Eq ((Algebra.trace R (Prod S T)) x) (HAdd.hAdd ((Algebra.trace R S) x.1) ((A …
  -/
  simp_rw [trace, this]
  /-
    R : Type u_1
    S : Type u_2
    T : Type u_3
    inst✝⁸ : CommRing R
    inst✝⁷ : CommRing S
    inst✝⁶ : CommRing T
    inst✝⁵ : Algebra R S
    inst✝⁴ : Algebra R T
    inst✝³ : Module.Free R S
    inst✝² : Module.Free R T
    inst✝¹ : Module.Finite R S
    inst✝ : Module.Finite R T
    x : Prod S T
    a✝ : Nontrivial R
    f : LinearMap (RingHom.id R) (Prod S T) (Prod (Module.End R S) (Module.End R T …
    this : Eq (Algebra.lmul R (Prod S T)).toLinearMap ((LinearMap.prodMapLinear R  …
    ⊢ Eq (((LinearMap.trace R (Prod S T)).comp ((LinearMap.prodMapLinear R S T S T …
  -/
  exact trace_prodMap' _ _
  /-
    🎉 no goals
  -/


theorem trace_prod [Module.Free R S] [Module.Free R T] [Module.Finite R S] [Module.Finite R T] :
    trace R (S × T) = (trace R S).coprod (trace R T) :=
                            /-
                              R : Type u_1
                              S : Type u_2
                              T : Type u_3
                              inst✝⁸ : CommRing R
                              inst✝⁷ : CommRing S
                              inst✝⁶ : CommRing T
                              inst✝⁵ : Algebra R S
                              inst✝⁴ : Algebra R T
                              inst✝³ : Module.Free R S
                              inst✝² : Module.Free R T
                              inst✝¹ : Module.Finite R S
                              inst✝ : Module.Finite R T
                              p : Prod S T
                              ⊢ Eq ((Algebra.trace R (Prod S T)) p) (((Algebra.trace R S).coprod (Algebra.tr …
                            -/
  LinearMap.ext fun p => by rw [coprod_apply, trace_prod_apply]
                            /-
                              🎉 no goals
                            -/


/-- The `traceForm` maps `x y : S` to the trace of `x * y`.
It is a symmetric bilinear form and is nondegenerate if the extension is separable.-/
@[stacks 0BIK "Trace pairing"]
noncomputable def traceForm : BilinForm R S :=
  LinearMap.compr₂ (lmul R S).toLinearMap (trace R S)


@[simp]
theorem traceForm_apply (x y : S) : traceForm R S x y = trace R S (x * y) :=
  rfl


theorem traceForm_isSymm : (traceForm R S).IsSymm := fun _ _ => congr_arg (trace R S) (mul_comm _ _)


theorem traceForm_toMatrix [DecidableEq ι] (b : Basis ι R S) (i j) :
    BilinForm.toMatrix b (traceForm R S) i j = trace R S (b i * b j) := by
  /-
    R : Type u_1
    S : Type u_2
    inst✝⁴ : CommRing R
    inst✝³ : CommRing S
    inst✝² : Algebra R S
    ι : Type w
    inst✝¹ : Fintype ι
    inst✝ : DecidableEq ι
    b : Basis ι R S
    i j : ι
    ⊢ Eq ((BilinForm.toMatrix b) (Algebra.traceForm R S) i j) ((Algebra.trace R S) …
  -/
  rw [BilinForm.toMatrix_apply, traceForm_apply]
  /-
    🎉 no goals
  -/


