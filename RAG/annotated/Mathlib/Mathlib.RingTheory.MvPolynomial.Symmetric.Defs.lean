/-- The `n`th elementary symmetric function evaluated at the elements of `s` -/
def esymm (s : Multiset R) (n : ℕ) : R :=
  ((s.powersetCard n).map Multiset.prod).sum


theorem _root_.Finset.esymm_map_val {σ} (f : σ → R) (s : Finset σ) (n : ℕ) :
    (s.val.map f).esymm n = (s.powersetCard n).sum fun t => t.prod f := by
  /-
    R : Type u_1
    inst✝ : CommSemiring R
    σ : Type u_2
    f : σ → R
    s : Finset σ
    n : Nat
    ⊢ Eq ((Multiset.map f s.val).esymm n) ((Finset.powersetCard n s).sum fun t =>  …
  -/
  simp only [esymm, powersetCard_map, ← Finset.map_val_val_powersetCard, map_map]
  /-
    R : Type u_1
    inst✝ : CommSemiring R
    σ : Type u_2
    f : σ → R
    s : Finset σ
    n : Nat
    ⊢ Eq (Multiset.map (Function.comp (fun x => x.prod) (Function.comp (fun x => M …
  -/
  rfl
  /-
    🎉 no goals
  -/


lemma pow_smul_esymm {S : Type*} [Monoid S] [DistribMulAction S R] [IsScalarTower S R R]
    [SMulCommClass S R R] (s : S) (n : ℕ) (m : Multiset R) :
    s ^ n • m.esymm n = (m.map (s • ·)).esymm n := by
  /-
    R : Type u_1
    inst✝⁴ : CommSemiring R
    S : Type u_2
    inst✝³ : Monoid S
    inst✝² : DistribMulAction S R
    inst✝¹ : IsScalarTower S R R
    inst✝ : SMulCommClass S R R
    s : S
    n : Nat
    m : Multiset R
    ⊢ Eq (HSMul.hSMul (HPow.hPow s n) (m.esymm n)) ((Multiset.map (fun x => HSMul. …
  -/
  rw [esymm, smul_sum, map_map]
  /-
    R : Type u_1
    inst✝⁴ : CommSemiring R
    S : Type u_2
    inst✝³ : Monoid S
    inst✝² : DistribMulAction S R
    inst✝¹ : IsScalarTower S R R
    inst✝ : SMulCommClass S R R
    s : S
    n : Nat
    m : Multiset R
    ⊢ Eq (Multiset.map (Function.comp (fun x => HSMul.hSMul (HPow.hPow s n) x) Mul …
  -/
  trans ((powersetCard n m).map (fun x : Multiset R ↦ s ^ card x • x.prod)).sum
    /-
      R : Type u_1
      inst✝⁴ : CommSemiring R
      S : Type u_2
      inst✝³ : Monoid S
      inst✝² : DistribMulAction S R
      inst✝¹ : IsScalarTower S R R
      inst✝ : SMulCommClass S R R
      s : S
      n : Nat
      m : Multiset R
      ⊢ Eq (Multiset.map (Function.comp (fun x => HSMul.hSMul (HPow.hPow s n) x) Mul …
    -/
  · refine congr_arg _ (map_congr rfl (fun x hx ↦ ?_))
    /-
      R : Type u_1
      inst✝⁴ : CommSemiring R
      S : Type u_2
      inst✝³ : Monoid S
      inst✝² : DistribMulAction S R
      inst✝¹ : IsScalarTower S R R
      inst✝ : SMulCommClass S R R
      s : S
      n : Nat
      m x : Multiset R
      hx : Membership.mem (Multiset.powersetCard n m) x
      ⊢ Eq (Function.comp (fun x => HSMul.hSMul (HPow.hPow s n) x) Multiset.prod x)  …
    -/
    rw [Function.comp_apply, (mem_powersetCard.1 hx).2]
    /-
      🎉 no goals
    -/
    /-
      R : Type u_1
      inst✝⁴ : CommSemiring R
      S : Type u_2
      inst✝³ : Monoid S
      inst✝² : DistribMulAction S R
      inst✝¹ : IsScalarTower S R R
      inst✝ : SMulCommClass S R R
      s : S
      n : Nat
      m : Multiset R
      ⊢ Eq (Multiset.map (fun x => HSMul.hSMul (HPow.hPow s x.card) x.prod) (Multise …
    -/
  · simp_rw [smul_prod, esymm, powersetCard_map, map_map, Function.comp_def]
    /-
      🎉 no goals
    -/


/-- A `MvPolynomial φ` is symmetric if it is invariant under
permutations of its variables by the `rename` operation -/
def IsSymmetric [CommSemiring R] (φ : MvPolynomial σ R) : Prop :=
  ∀ e : Perm σ, rename e φ = φ


/-- The subalgebra of symmetric `MvPolynomial`s. -/
def symmetricSubalgebra (σ R : Type*) [CommSemiring R] : Subalgebra R (MvPolynomial σ R) where
  carrier := setOf IsSymmetric
  algebraMap_mem' r e := rename_C e r
                         /-
                           σ✝ : Type u_1
                           τ : Type u_2
                           R✝ : Type u_3
                           S : Type u_4
                           σ : Type u_5
                           R : Type u_6
                           inst✝ : CommSemiring R
                           a✝ b✝ : MvPolynomial σ R
                           ha : Membership.mem (setOf MvPolynomial.IsSymmetric) a✝
                           hb : Membership.mem (setOf MvPolynomial.IsSymmetric) b✝
                           e : Equiv.Perm σ
                           ⊢ Eq ((MvPolynomial.rename ⇑e) (HMul.hMul a✝ b✝)) (HMul.hMul a✝ b✝)
                         -/
  mul_mem' ha hb e := by rw [map_mul, ha, hb]
                         /-
                           🎉 no goals
                         -/
                         /-
                           σ✝ : Type u_1
                           τ : Type u_2
                           R✝ : Type u_3
                           S : Type u_4
                           σ : Type u_5
                           R : Type u_6
                           inst✝ : CommSemiring R
                           a✝ b✝ : MvPolynomial σ R
                           ha : Membership.mem { carrier := setOf MvPolynomial.IsSymmetric, mul_mem' := ⋯ …
                           hb : Membership.mem { carrier := setOf MvPolynomial.IsSymmetric, mul_mem' := ⋯ …
                           e : Equiv.Perm σ
                           ⊢ Eq ((MvPolynomial.rename ⇑e) (HAdd.hAdd a✝ b✝)) (HAdd.hAdd a✝ b✝)
                         -/
  add_mem' ha hb e := by rw [map_add, ha, hb]
                         /-
                           🎉 no goals
                         -/


@[simp]
theorem mem_symmetricSubalgebra [CommSemiring R] (p : MvPolynomial σ R) :
    p ∈ symmetricSubalgebra σ R ↔ p.IsSymmetric :=
  Iff.rfl


@[simp]
theorem C (r : R) : IsSymmetric (C r : MvPolynomial σ R) :=
  (symmetricSubalgebra σ R).algebraMap_mem r


@[simp]
theorem zero : IsSymmetric (0 : MvPolynomial σ R) :=
  (symmetricSubalgebra σ R).zero_mem


@[simp]
theorem one : IsSymmetric (1 : MvPolynomial σ R) :=
  (symmetricSubalgebra σ R).one_mem


theorem add (hφ : IsSymmetric φ) (hψ : IsSymmetric ψ) : IsSymmetric (φ + ψ) :=
  (symmetricSubalgebra σ R).add_mem hφ hψ


theorem mul (hφ : IsSymmetric φ) (hψ : IsSymmetric ψ) : IsSymmetric (φ * ψ) :=
  (symmetricSubalgebra σ R).mul_mem hφ hψ


theorem smul (r : R) (hφ : IsSymmetric φ) : IsSymmetric (r • φ) :=
  (symmetricSubalgebra σ R).smul_mem hφ r


@[simp]
theorem map (hφ : IsSymmetric φ) (f : R →+* S) : IsSymmetric (map f φ) := fun e => by
  /-
    σ : Type u_1
    R : Type u_3
    S : Type u_4
    inst✝¹ : CommSemiring R
    inst✝ : CommSemiring S
    φ : MvPolynomial σ R
    hφ : φ.IsSymmetric
    f : RingHom R S
    e : Equiv.Perm σ
    ⊢ Eq ((MvPolynomial.rename ⇑e) ((MvPolynomial.map f) φ)) ((MvPolynomial.map f) …
  -/
  rw [← map_rename, hφ]
  /-
    🎉 no goals
  -/


protected theorem rename (hφ : φ.IsSymmetric) (e : σ ≃ τ) : (rename e φ).IsSymmetric := fun _ => by
  /-
    σ : Type u_1
    τ : Type u_2
    R : Type u_3
    inst✝ : CommSemiring R
    φ : MvPolynomial σ R
    hφ : φ.IsSymmetric
    e : Equiv σ τ
    x✝ : Equiv.Perm τ
    ⊢ Eq ((MvPolynomial.rename ⇑x✝) ((MvPolynomial.rename ⇑e) φ)) ((MvPolynomial.r …
  -/
  apply rename_injective _ e.symm.injective
  /-
    case a
    σ : Type u_1
    τ : Type u_2
    R : Type u_3
    inst✝ : CommSemiring R
    φ : MvPolynomial σ R
    hφ : φ.IsSymmetric
    e : Equiv σ τ
    x✝ : Equiv.Perm τ
    ⊢ Eq ((MvPolynomial.rename ⇑e.symm) ((MvPolynomial.rename ⇑x✝) ((MvPolynomial. …
  -/
  simp_rw [rename_rename, ← Equiv.coe_trans, Equiv.self_trans_symm, Equiv.coe_refl, rename_id]
  /-
    case a
    σ : Type u_1
    τ : Type u_2
    R : Type u_3
    inst✝ : CommSemiring R
    φ : MvPolynomial σ R
    hφ : φ.IsSymmetric
    e : Equiv σ τ
    x✝ : Equiv.Perm τ
    ⊢ Eq ((MvPolynomial.rename ⇑((e.trans x✝).trans e.symm)) φ) φ
  -/
  rw [hφ]
  /-
    🎉 no goals
  -/


@[simp]
theorem _root_.MvPolynomial.isSymmetric_rename {e : σ ≃ τ} :
    (MvPolynomial.rename e φ).IsSymmetric ↔ φ.IsSymmetric :=
               /-
                 σ : Type u_1
                 τ : Type u_2
                 R : Type u_3
                 inst✝ : CommSemiring R
                 φ : MvPolynomial σ R
                 e : Equiv σ τ
                 h : ((MvPolynomial.rename ⇑e) φ).IsSymmetric
                 ⊢ φ.IsSymmetric
               -/
  ⟨fun h => by simpa using (IsSymmetric.rename (R := R) h e.symm), (IsSymmetric.rename · e)⟩
               /-
                 🎉 no goals
               -/


theorem neg (hφ : IsSymmetric φ) : IsSymmetric (-φ) :=
  (symmetricSubalgebra σ R).neg_mem hφ


theorem sub (hφ : IsSymmetric φ) (hψ : IsSymmetric ψ) : IsSymmetric (φ - ψ) :=
  (symmetricSubalgebra σ R).sub_mem hφ hψ


/-- `MvPolynomial.rename` induces an isomorphism between the symmetric subalgebras. -/
@[simps!]
def renameSymmetricSubalgebra [CommSemiring R] (e : σ ≃ τ) :
    symmetricSubalgebra σ R ≃ₐ[R] symmetricSubalgebra τ R :=
  AlgEquiv.ofAlgHom
    (((rename e).comp (symmetricSubalgebra σ R).val).codRestrict _ <| fun x => x.2.rename e)
    (((rename e.symm).comp <| Subalgebra.val _).codRestrict _ <| fun x => x.2.rename e.symm)
                                              /-
                                                σ : Type u_1
                                                τ : Type u_2
                                                R : Type u_3
                                                S : Type u_4
                                                inst✝ : CommSemiring R
                                                e : Equiv σ τ
                                                p : Subtype fun x => Membership.mem (MvPolynomial.symmetricSubalgebra τ R) x
                                                ⊢ Eq ↑(((((MvPolynomial.rename ⇑e).comp (MvPolynomial.symmetricSubalgebra σ R) …
                                              -/
    (AlgHom.ext <| fun p => Subtype.ext <| by simp)
                                              /-
                                                🎉 no goals
                                              -/
                                              /-
                                                σ : Type u_1
                                                τ : Type u_2
                                                R : Type u_3
                                                S : Type u_4
                                                inst✝ : CommSemiring R
                                                e : Equiv σ τ
                                                p : Subtype fun x => Membership.mem (MvPolynomial.symmetricSubalgebra σ R) x
                                                ⊢ Eq ↑(((((MvPolynomial.rename ⇑e.symm).comp (MvPolynomial.symmetricSubalgebra …
                                              -/
    (AlgHom.ext <| fun p => Subtype.ext <| by simp)
                                              /-
                                                🎉 no goals
                                              -/


/-- The `n`th elementary symmetric `MvPolynomial σ R`.
It is the sum over all the degree n squarefree monomials in `MvPolynomial σ R`. -/
def esymm (n : ℕ) : MvPolynomial σ R :=
  ∑ t ∈ powersetCard n univ, ∏ i ∈ t, X i


/--
`esymmPart` is the product of the symmetric polynomials `esymm μᵢ`,
where `μ = (μ₁, μ₂, ...)` is a partition.
-/
def esymmPart {n : ℕ} (μ : n.Partition) : MvPolynomial σ R := (μ.parts.map (esymm σ R)).prod


/-- The `n`th elementary symmetric `MvPolynomial σ R` is obtained by evaluating the
`n`th elementary symmetric at the `Multiset` of the monomials -/
theorem esymm_eq_multiset_esymm : esymm σ R = (univ.val.map X).esymm := by
  /-
    σ : Type u_5
    R : Type u_6
    inst✝¹ : CommSemiring R
    inst✝ : Fintype σ
    ⊢ Eq (MvPolynomial.esymm σ R) (Multiset.map MvPolynomial.X Finset.univ.val).es …
  -/
  exact funext fun n => (esymm_map_val X _ n).symm
  /-
    🎉 no goals
  -/


theorem aeval_esymm_eq_multiset_esymm [Algebra R S] (n : ℕ) (f : σ → S) :
    aeval f (esymm σ R n) = (univ.val.map f).esymm n := by
  /-
    S : Type u_4
    σ : Type u_5
    R : Type u_6
    inst✝³ : CommSemiring R
    inst✝² : CommSemiring S
    inst✝¹ : Fintype σ
    inst✝ : Algebra R S
    n : Nat
    f : σ → S
    ⊢ Eq ((MvPolynomial.aeval f) (MvPolynomial.esymm σ R n)) ((Multiset.map f Fins …
  -/
  simp_rw [esymm, aeval_sum, aeval_prod, aeval_X, esymm_map_val]
  /-
    🎉 no goals
  -/


/-- We can define `esymm σ R n` by summing over a subtype instead of over `powerset_len`. -/
theorem esymm_eq_sum_subtype (n : ℕ) :
    esymm σ R n = ∑ t : {s : Finset σ // #s = n}, ∏ i ∈ (t : Finset σ), X i :=
  sum_subtype _ (fun _ => mem_powersetCard_univ) _


/-- We can define `esymm σ R n` as a sum over explicit monomials -/
theorem esymm_eq_sum_monomial (n : ℕ) :
    esymm σ R n = ∑ t ∈ powersetCard n univ, monomial (∑ i ∈ t, Finsupp.single i 1) 1 := by
  /-
    σ : Type u_5
    R : Type u_6
    inst✝¹ : CommSemiring R
    inst✝ : Fintype σ
    n : Nat
    ⊢ Eq (MvPolynomial.esymm σ R n) ((Finset.powersetCard n Finset.univ).sum fun t …
  -/
  simp_rw [monomial_sum_one]
  /-
    σ : Type u_5
    R : Type u_6
    inst✝¹ : CommSemiring R
    inst✝ : Fintype σ
    n : Nat
    ⊢ Eq (MvPolynomial.esymm σ R n) ((Finset.powersetCard n Finset.univ).sum fun x …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem esymm_zero : esymm σ R 0 = 1 := by
  /-
    σ : Type u_5
    R : Type u_6
    inst✝¹ : CommSemiring R
    inst✝ : Fintype σ
    ⊢ Eq (MvPolynomial.esymm σ R 0) 1
  -/
  simp only [esymm, powersetCard_zero, sum_singleton, prod_empty]
  /-
    🎉 no goals
  -/


@[simp]
                                                 /-
                                                   σ : Type u_5
                                                   R : Type u_6
                                                   inst✝¹ : CommSemiring R
                                                   inst✝ : Fintype σ
                                                   ⊢ Eq (MvPolynomial.esymm σ R 1) (Finset.univ.sum fun i => MvPolynomial.X i)
                                                 -/
theorem esymm_one : esymm σ R 1 = ∑ i, X i := by simp [esymm, powersetCard_one]
                                                 /-
                                                   🎉 no goals
                                                 -/


                                                                 /-
                                                                   σ : Type u_5
                                                                   R : Type u_6
                                                                   inst✝¹ : CommSemiring R
                                                                   inst✝ : Fintype σ
                                                                   ⊢ Eq (MvPolynomial.esymmPart σ R (Nat.Partition.indiscrete 0)) 1
                                                                 -/
theorem esymmPart_zero : esymmPart σ R (.indiscrete 0) = 1 := by simp [esymmPart]
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


@[simp]
theorem esymmPart_indiscrete (n : ℕ) : esymmPart σ R (.indiscrete n) = esymm σ R n := by
  /-
    σ : Type u_5
    R : Type u_6
    inst✝¹ : CommSemiring R
    inst✝ : Fintype σ
    n : Nat
    ⊢ Eq (MvPolynomial.esymmPart σ R (Nat.Partition.indiscrete n)) (MvPolynomial.e …
  -/
              /-
                🎉 no goals
              -/
  cases n <;> simp [esymmPart]
              /-
                🎉 no goals
              -/


theorem map_esymm (n : ℕ) (f : R →+* S) : map f (esymm σ R n) = esymm σ S n := by
  /-
    S : Type u_4
    σ : Type u_5
    R : Type u_6
    inst✝² : CommSemiring R
    inst✝¹ : CommSemiring S
    inst✝ : Fintype σ
    n : Nat
    f : RingHom R S
    ⊢ Eq ((MvPolynomial.map f) (MvPolynomial.esymm σ R n)) (MvPolynomial.esymm σ S …
  -/
  simp_rw [esymm, map_sum, map_prod, map_X]
  /-
    🎉 no goals
  -/


theorem rename_esymm (n : ℕ) (e : σ ≃ τ) : rename e (esymm σ R n) = esymm τ R n :=
  calc
    rename e (esymm σ R n) = ∑ x ∈ powersetCard n univ, ∏ i ∈ x, X (e i) := by
      /-
        τ : Type u_2
        σ : Type u_5
        R : Type u_6
        inst✝² : CommSemiring R
        inst✝¹ : Fintype σ
        inst✝ : Fintype τ
        n : Nat
        e : Equiv σ τ
        ⊢ Eq ((MvPolynomial.rename ⇑e) (MvPolynomial.esymm σ R n)) ((Finset.powersetCa …
      -/
      simp_rw [esymm, map_sum, map_prod, rename_X]
      /-
        🎉 no goals
      -/
    _ = ∑ t ∈ powersetCard n (univ.map e.toEmbedding), ∏ i ∈ t, X i := by
      /-
        τ : Type u_2
        σ : Type u_5
        R : Type u_6
        inst✝² : CommSemiring R
        inst✝¹ : Fintype σ
        inst✝ : Fintype τ
        n : Nat
        e : Equiv σ τ
        ⊢ Eq ((Finset.powersetCard n Finset.univ).sum fun x => x.prod fun i => MvPolyn …
      -/
      simp [powersetCard_map, -map_univ_equiv]
      -- Porting note: Why did `mapEmbedding_apply` not work?
      /-
        τ : Type u_2
        σ : Type u_5
        R : Type u_6
        inst✝² : CommSemiring R
        inst✝¹ : Fintype σ
        inst✝ : Fintype τ
        n : Nat
        e : Equiv σ τ
        ⊢ Eq ((Finset.powersetCard n Finset.univ).sum fun x => x.prod fun i => MvPolyn …
      -/
      dsimp [mapEmbedding, OrderEmbedding.ofMapLEIff]
      /-
        τ : Type u_2
        σ : Type u_5
        R : Type u_6
        inst✝² : CommSemiring R
        inst✝¹ : Fintype σ
        inst✝ : Fintype τ
        n : Nat
        e : Equiv σ τ
        ⊢ Eq ((Finset.powersetCard n Finset.univ).sum fun x => x.prod fun i => MvPolyn …
      -/
      simp
      /-
        🎉 no goals
      -/
                                                      /-
                                                        τ : Type u_2
                                                        σ : Type u_5
                                                        R : Type u_6
                                                        inst✝² : CommSemiring R
                                                        inst✝¹ : Fintype σ
                                                        inst✝ : Fintype τ
                                                        n : Nat
                                                        e : Equiv σ τ
                                                        ⊢ Eq ((Finset.powersetCard n (Finset.map e.toEmbedding Finset.univ)).sum fun t …
                                                      -/
    _ = ∑ t ∈ powersetCard n univ, ∏ i ∈ t, X i := by rw [map_univ_equiv]
                                                      /-
                                                        🎉 no goals
                                                      -/


theorem esymm_isSymmetric (n : ℕ) : IsSymmetric (esymm σ R n) := by
  /-
    σ : Type u_5
    R : Type u_6
    inst✝¹ : CommSemiring R
    inst✝ : Fintype σ
    n : Nat
    ⊢ (MvPolynomial.esymm σ R n).IsSymmetric
  -/
  intro
  /-
    σ : Type u_5
    R : Type u_6
    inst✝¹ : CommSemiring R
    inst✝ : Fintype σ
    n : Nat
    e✝ : Equiv.Perm σ
    ⊢ Eq ((MvPolynomial.rename ⇑e✝) (MvPolynomial.esymm σ R n)) (MvPolynomial.esym …
  -/
  rw [rename_esymm]
  /-
    🎉 no goals
  -/


theorem support_esymm'' [DecidableEq σ] [Nontrivial R] (n : ℕ) :
    (esymm σ R n).support =
      (powersetCard n (univ : Finset σ)).biUnion fun t =>
        (Finsupp.single (∑ i ∈ t, Finsupp.single i 1) (1 : R)).support := by
  /-
    σ : Type u_5
    R : Type u_6
    inst✝³ : CommSemiring R
    inst✝² : Fintype σ
    inst✝¹ : DecidableEq σ
    inst✝ : Nontrivial R
    n : Nat
    ⊢ Eq (MvPolynomial.esymm σ R n).support ((Finset.powersetCard n Finset.univ).b …
  -/
  rw [esymm_eq_sum_monomial]
  /-
    σ : Type u_5
    R : Type u_6
    inst✝³ : CommSemiring R
    inst✝² : Fintype σ
    inst✝¹ : DecidableEq σ
    inst✝ : Nontrivial R
    n : Nat
    ⊢ Eq ((Finset.powersetCard n Finset.univ).sum fun t => (MvPolynomial.monomial  …
  -/
  simp only [← single_eq_monomial]
  /-
    σ : Type u_5
    R : Type u_6
    inst✝³ : CommSemiring R
    inst✝² : Fintype σ
    inst✝¹ : DecidableEq σ
    inst✝ : Nontrivial R
    n : Nat
    ⊢ Eq ((Finset.powersetCard n Finset.univ).sum fun x => Finsupp.single (x.sum f …
  -/
  refine Finsupp.support_sum_eq_biUnion (powersetCard n (univ : Finset σ)) ?_
  /-
    σ : Type u_5
    R : Type u_6
    inst✝³ : CommSemiring R
    inst✝² : Fintype σ
    inst✝¹ : DecidableEq σ
    inst✝ : Nontrivial R
    n : Nat
    ⊢ ∀ (i₁ i₂ : Finset σ), Ne i₁ i₂ → Disjoint (Finsupp.single (i₁.sum fun i => F …
  -/
  intro s t hst
  /-
    σ : Type u_5
    R : Type u_6
    inst✝³ : CommSemiring R
    inst✝² : Fintype σ
    inst✝¹ : DecidableEq σ
    inst✝ : Nontrivial R
    n : Nat
    s t : Finset σ
    hst : Ne s t
    ⊢ Disjoint (Finsupp.single (s.sum fun i => Finsupp.single i 1) 1).support (Fin …
  -/
  rw [disjoint_left, Finsupp.support_single_ne_zero _ one_ne_zero]
  /-
    σ : Type u_5
    R : Type u_6
    inst✝³ : CommSemiring R
    inst✝² : Fintype σ
    inst✝¹ : DecidableEq σ
    inst✝ : Nontrivial R
    n : Nat
    s t : Finset σ
    hst : Ne s t
    ⊢ ∀ ⦃a : Finsupp σ Nat⦄, Membership.mem (Singleton.singleton (s.sum fun i => F …
  -/
  rw [Finsupp.support_single_ne_zero _ one_ne_zero]
  /-
    σ : Type u_5
    R : Type u_6
    inst✝³ : CommSemiring R
    inst✝² : Fintype σ
    inst✝¹ : DecidableEq σ
    inst✝ : Nontrivial R
    n : Nat
    s t : Finset σ
    hst : Ne s t
    ⊢ ∀ ⦃a : Finsupp σ Nat⦄, Membership.mem (Singleton.singleton (s.sum fun i => F …
  -/
  simp only [one_ne_zero, mem_singleton, Finsupp.mem_support_iff]
  /-
    σ : Type u_5
    R : Type u_6
    inst✝³ : CommSemiring R
    inst✝² : Fintype σ
    inst✝¹ : DecidableEq σ
    inst✝ : Nontrivial R
    n : Nat
    s t : Finset σ
    hst : Ne s t
    ⊢ ∀ ⦃a : Finsupp σ Nat⦄, Eq a (s.sum fun i => Finsupp.single i 1) → Not (Eq a  …
  -/
  rintro a h rfl
  /-
    σ : Type u_5
    R : Type u_6
    inst✝³ : CommSemiring R
    inst✝² : Fintype σ
    inst✝¹ : DecidableEq σ
    inst✝ : Nontrivial R
    n : Nat
    s t : Finset σ
    hst : Ne s t
    h : Eq (t.sum fun i => Finsupp.single i 1) (s.sum fun i => Finsupp.single i 1)
    ⊢ False
  -/
  have := congr_arg Finsupp.support h
  /-
    σ : Type u_5
    R : Type u_6
    inst✝³ : CommSemiring R
    inst✝² : Fintype σ
    inst✝¹ : DecidableEq σ
    inst✝ : Nontrivial R
    n : Nat
    s t : Finset σ
    hst : Ne s t
    h : Eq (t.sum fun i => Finsupp.single i 1) (s.sum fun i => Finsupp.single i 1)
    this : Eq (t.sum fun i => Finsupp.single i 1).support (s.sum fun i => Finsupp. …
    ⊢ False
  -/
  rw [Finsupp.support_sum_eq_biUnion, Finsupp.support_sum_eq_biUnion] at this
  · have hsingle : ∀ s : Finset σ, ∀ x : σ, x ∈ s → (Finsupp.single x 1).support = {x} := by
      intros _ x _
      rw [Finsupp.support_single_ne_zero x one_ne_zero]
    /-
      σ : Type u_5
      R : Type u_6
      inst✝³ : CommSemiring R
      inst✝² : Fintype σ
      inst✝¹ : DecidableEq σ
      inst✝ : Nontrivial R
      n : Nat
      s t : Finset σ
      hst : Ne s t
      h : Eq (t.sum fun i => Finsupp.single i 1) (s.sum fun i => Finsupp.single i 1)
      this : Eq (t.biUnion fun i => (Finsupp.single i 1).support) (s.biUnion fun i = …
      hsingle : ∀ (s : Finset σ) (x : σ), Membership.mem s x → Eq (Finsupp.single x  …
      ⊢ False
    -/
    have hs := biUnion_congr (of_eq_true (eq_self s)) (hsingle s)
    /-
      σ : Type u_5
      R : Type u_6
      inst✝³ : CommSemiring R
      inst✝² : Fintype σ
      inst✝¹ : DecidableEq σ
      inst✝ : Nontrivial R
      n : Nat
      s t : Finset σ
      hst : Ne s t
      h : Eq (t.sum fun i => Finsupp.single i 1) (s.sum fun i => Finsupp.single i 1)
      this : Eq (t.biUnion fun i => (Finsupp.single i 1).support) (s.biUnion fun i = …
      hsingle : ∀ (s : Finset σ) (x : σ), Membership.mem s x → Eq (Finsupp.single x  …
      hs : Eq (s.biUnion fun a => (Finsupp.single a 1).support) (s.biUnion Singleton …
      ⊢ False
    -/
    have ht := biUnion_congr (of_eq_true (eq_self t)) (hsingle t)
    /-
      σ : Type u_5
      R : Type u_6
      inst✝³ : CommSemiring R
      inst✝² : Fintype σ
      inst✝¹ : DecidableEq σ
      inst✝ : Nontrivial R
      n : Nat
      s t : Finset σ
      hst : Ne s t
      h : Eq (t.sum fun i => Finsupp.single i 1) (s.sum fun i => Finsupp.single i 1)
      this : Eq (t.biUnion fun i => (Finsupp.single i 1).support) (s.biUnion fun i = …
      hsingle : ∀ (s : Finset σ) (x : σ), Membership.mem s x → Eq (Finsupp.single x  …
      hs : Eq (s.biUnion fun a => (Finsupp.single a 1).support) (s.biUnion Singleton …
      ht : Eq (t.biUnion fun a => (Finsupp.single a 1).support) (t.biUnion Singleton …
      ⊢ False
    -/
    rw [hs, ht] at this
      /-
        σ : Type u_5
        R : Type u_6
        inst✝³ : CommSemiring R
        inst✝² : Fintype σ
        inst✝¹ : DecidableEq σ
        inst✝ : Nontrivial R
        n : Nat
        s t : Finset σ
        hst : Ne s t
        h : Eq (t.sum fun i => Finsupp.single i 1) (s.sum fun i => Finsupp.single i 1)
        this : Eq (t.biUnion Singleton.singleton) (s.biUnion Singleton.singleton)
        hsingle : ∀ (s : Finset σ) (x : σ), Membership.mem s x → Eq (Finsupp.single x  …
        hs : Eq (s.biUnion fun a => (Finsupp.single a 1).support) (s.biUnion Singleton …
        ht : Eq (t.biUnion fun a => (Finsupp.single a 1).support) (t.biUnion Singleton …
        ⊢ False
      -/
    · simp only [biUnion_singleton_eq_self] at this
      /-
        σ : Type u_5
        R : Type u_6
        inst✝³ : CommSemiring R
        inst✝² : Fintype σ
        inst✝¹ : DecidableEq σ
        inst✝ : Nontrivial R
        n : Nat
        s t : Finset σ
        hst : Ne s t
        h : Eq (t.sum fun i => Finsupp.single i 1) (s.sum fun i => Finsupp.single i 1)
        hsingle : ∀ (s : Finset σ) (x : σ), Membership.mem s x → Eq (Finsupp.single x  …
        hs : Eq (s.biUnion fun a => (Finsupp.single a 1).support) (s.biUnion Singleton …
        ht : Eq (t.biUnion fun a => (Finsupp.single a 1).support) (t.biUnion Singleton …
        this : Eq t s
        ⊢ False
      -/
      exact absurd this hst.symm
      /-
        🎉 no goals
      -/
  /-
    case h
    σ : Type u_5
    R : Type u_6
    inst✝³ : CommSemiring R
    inst✝² : Fintype σ
    inst✝¹ : DecidableEq σ
    inst✝ : Nontrivial R
    n : Nat
    s t : Finset σ
    hst : Ne s t
    h : Eq (t.sum fun i => Finsupp.single i 1) (s.sum fun i => Finsupp.single i 1)
    this : Eq (t.biUnion fun i => (Finsupp.single i 1).support) (s.sum fun i => Fi …
    ⊢ ∀ (i₁ i₂ : σ), Ne i₁ i₂ → Disjoint (Finsupp.single i₁ 1).support (Finsupp.si …
  -/
  all_goals intro x y; simp [Finsupp.support_single_disjoint]
  /-
    🎉 no goals
  -/


theorem support_esymm' [DecidableEq σ] [Nontrivial R] (n : ℕ) : (esymm σ R n).support =
    (powersetCard n (univ : Finset σ)).biUnion fun t => {∑ i ∈ t, Finsupp.single i 1} := by
  /-
    σ : Type u_5
    R : Type u_6
    inst✝³ : CommSemiring R
    inst✝² : Fintype σ
    inst✝¹ : DecidableEq σ
    inst✝ : Nontrivial R
    n : Nat
    ⊢ Eq (MvPolynomial.esymm σ R n).support ((Finset.powersetCard n Finset.univ).b …
  -/
  rw [support_esymm'']
  /-
    σ : Type u_5
    R : Type u_6
    inst✝³ : CommSemiring R
    inst✝² : Fintype σ
    inst✝¹ : DecidableEq σ
    inst✝ : Nontrivial R
    n : Nat
    ⊢ Eq ((Finset.powersetCard n Finset.univ).biUnion fun t => (Finsupp.single (t. …
  -/
  congr
  /-
    case e_t
    σ : Type u_5
    R : Type u_6
    inst✝³ : CommSemiring R
    inst✝² : Fintype σ
    inst✝¹ : DecidableEq σ
    inst✝ : Nontrivial R
    n : Nat
    ⊢ Eq (fun t => (Finsupp.single (t.sum fun i => Finsupp.single i 1) 1).support) …
  -/
  funext
  /-
    case e_t.h
    σ : Type u_5
    R : Type u_6
    inst✝³ : CommSemiring R
    inst✝² : Fintype σ
    inst✝¹ : DecidableEq σ
    inst✝ : Nontrivial R
    n : Nat
    x✝ : Finset σ
    ⊢ Eq (Finsupp.single (x✝.sum fun i => Finsupp.single i 1) 1).support (Singleto …
  -/
  exact Finsupp.support_single_ne_zero _ one_ne_zero
  /-
    🎉 no goals
  -/


theorem support_esymm [DecidableEq σ] [Nontrivial R] (n : ℕ) : (esymm σ R n).support =
    (powersetCard n (univ : Finset σ)).image fun t => ∑ i ∈ t, Finsupp.single i 1 := by
  /-
    σ : Type u_5
    R : Type u_6
    inst✝³ : CommSemiring R
    inst✝² : Fintype σ
    inst✝¹ : DecidableEq σ
    inst✝ : Nontrivial R
    n : Nat
    ⊢ Eq (MvPolynomial.esymm σ R n).support (Finset.image (fun t => t.sum fun i => …
  -/
  rw [support_esymm']
  /-
    σ : Type u_5
    R : Type u_6
    inst✝³ : CommSemiring R
    inst✝² : Fintype σ
    inst✝¹ : DecidableEq σ
    inst✝ : Nontrivial R
    n : Nat
    ⊢ Eq ((Finset.powersetCard n Finset.univ).biUnion fun t => Singleton.singleton …
  -/
  exact biUnion_singleton
  /-
    🎉 no goals
  -/


theorem degrees_esymm [Nontrivial R] {n : ℕ} (hpos : 0 < n) (hn : n ≤ Fintype.card σ) :
    (esymm σ R n).degrees = (univ : Finset σ).val := by
  classical
    have :
      (Finsupp.toMultiset ∘ fun t : Finset σ => ∑ i ∈ t, Finsupp.single i 1) = val := by
      funext
      simp [Finsupp.toMultiset_sum_single]
    rw [degrees_def, support_esymm, sup_image, this]
    have : ((powersetCard n univ).sup (fun (x : Finset σ) => x)).val
        = sup (powersetCard n univ) val := by
      refine comp_sup_eq_sup_comp _ ?_ ?_
      · intros
        simp only [union_val, sup_eq_union]
        congr
      · rfl
    rw [← this]
    obtain ⟨k, rfl⟩ := Nat.exists_eq_succ_of_ne_zero hpos.ne'
    simpa using powersetCard_sup _ _ (Nat.lt_of_succ_le hn)


/-- The `n`th complete homogeneous symmetric `MvPolynomial σ R`.
It is the sum over all the degree n monomials in `MvPolynomial σ R`. -/
def hsymm (n : ℕ) : MvPolynomial σ R := ∑ s : Sym σ n, (s.1.map X).prod


/-- `hsymmPart` is the product of the symmetric polynomials `hsymm μᵢ`,
where `μ = (μ₁, μ₂, ...)` is a partition. -/
def hsymmPart {n : ℕ} (μ : n.Partition) : MvPolynomial σ R := (μ.parts.map (hsymm σ R)).prod


@[simp]
                                           /-
                                             σ : Type u_5
                                             R : Type u_6
                                             inst✝² : CommSemiring R
                                             inst✝¹ : Fintype σ
                                             inst✝ : DecidableEq σ
                                             ⊢ Eq (MvPolynomial.hsymm σ R 0) 1
                                           -/
theorem hsymm_zero : hsymm σ R 0 = 1 := by simp [hsymm, eq_nil_of_card_zero]
                                           /-
                                             🎉 no goals
                                           -/


@[simp]
theorem hsymm_one : hsymm σ R 1 = ∑ i, X i := by
  /-
    σ : Type u_5
    R : Type u_6
    inst✝² : CommSemiring R
    inst✝¹ : Fintype σ
    inst✝ : DecidableEq σ
    ⊢ Eq (MvPolynomial.hsymm σ R 1) (Finset.univ.sum fun i => MvPolynomial.X i)
  -/
  symm
  /-
    σ : Type u_5
    R : Type u_6
    inst✝² : CommSemiring R
    inst✝¹ : Fintype σ
    inst✝ : DecidableEq σ
    ⊢ Eq (Finset.univ.sum fun i => MvPolynomial.X i) (MvPolynomial.hsymm σ R 1)
  -/
  apply Fintype.sum_equiv oneEquiv
  /-
    case h
    σ : Type u_5
    R : Type u_6
    inst✝² : CommSemiring R
    inst✝¹ : Fintype σ
    inst✝ : DecidableEq σ
    ⊢ ∀ (x : σ), Eq (MvPolynomial.X x) (Multiset.map MvPolynomial.X ↑(Sym.oneEquiv …
  -/
  simp only [oneEquiv_apply, Multiset.map_singleton, Multiset.prod_singleton, implies_true]
  /-
    🎉 no goals
  -/


                                                                 /-
                                                                   σ : Type u_5
                                                                   R : Type u_6
                                                                   inst✝² : CommSemiring R
                                                                   inst✝¹ : Fintype σ
                                                                   inst✝ : DecidableEq σ
                                                                   ⊢ Eq (MvPolynomial.hsymmPart σ R (Nat.Partition.indiscrete 0)) 1
                                                                 -/
theorem hsymmPart_zero : hsymmPart σ R (.indiscrete 0) = 1 := by simp [hsymmPart]
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


@[simp]
theorem hsymmPart_indiscrete (n : ℕ) : hsymmPart σ R (.indiscrete n) = hsymm σ R n := by
  /-
    σ : Type u_5
    R : Type u_6
    inst✝² : CommSemiring R
    inst✝¹ : Fintype σ
    inst✝ : DecidableEq σ
    n : Nat
    ⊢ Eq (MvPolynomial.hsymmPart σ R (Nat.Partition.indiscrete n)) (MvPolynomial.h …
  -/
              /-
                🎉 no goals
              -/
  cases n <;> simp [hsymmPart]
              /-
                🎉 no goals
              -/


theorem map_hsymm (n : ℕ) (f : R →+* S) : map f (hsymm σ R n) = hsymm σ S n := by
  /-
    S : Type u_4
    σ : Type u_5
    R : Type u_6
    inst✝³ : CommSemiring R
    inst✝² : CommSemiring S
    inst✝¹ : Fintype σ
    inst✝ : DecidableEq σ
    n : Nat
    f : RingHom R S
    ⊢ Eq ((MvPolynomial.map f) (MvPolynomial.hsymm σ R n)) (MvPolynomial.hsymm σ S …
  -/
  simp [hsymm, ← Multiset.prod_hom']
  /-
    🎉 no goals
  -/


theorem rename_hsymm (n : ℕ) (e : σ ≃ τ) : rename e (hsymm σ R n) = hsymm τ R n := by
  /-
    τ : Type u_2
    σ : Type u_5
    R : Type u_6
    inst✝⁴ : CommSemiring R
    inst✝³ : Fintype σ
    inst✝² : Fintype τ
    inst✝¹ : DecidableEq σ
    inst✝ : DecidableEq τ
    n : Nat
    e : Equiv σ τ
    ⊢ Eq ((MvPolynomial.rename ⇑e) (MvPolynomial.hsymm σ R n)) (MvPolynomial.hsymm …
  -/
  simp_rw [hsymm, map_sum, ← prod_hom', rename_X]
  /-
    τ : Type u_2
    σ : Type u_5
    R : Type u_6
    inst✝⁴ : CommSemiring R
    inst✝³ : Fintype σ
    inst✝² : Fintype τ
    inst✝¹ : DecidableEq σ
    inst✝ : DecidableEq τ
    n : Nat
    e : Equiv σ τ
    ⊢ Eq (Finset.univ.sum fun x => (Multiset.map (fun x => MvPolynomial.X (e x)) ↑ …
  -/
  apply Fintype.sum_equiv (equivCongr e)
  /-
    case h
    τ : Type u_2
    σ : Type u_5
    R : Type u_6
    inst✝⁴ : CommSemiring R
    inst✝³ : Fintype σ
    inst✝² : Fintype τ
    inst✝¹ : DecidableEq σ
    inst✝ : DecidableEq τ
    n : Nat
    e : Equiv σ τ
    ⊢ ∀ (x : Sym σ n), Eq (Multiset.map (fun x => MvPolynomial.X (e x)) ↑x).prod ( …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem hsymm_isSymmetric (n : ℕ) : IsSymmetric (hsymm σ R n) := rename_hsymm _ _ n


/-- The degree-`n` power sum symmetric `MvPolynomial σ R`.
It is the sum over all the `n`-th powers of the variables. -/
def psum (n : ℕ) : MvPolynomial σ R := ∑ i, X i ^ n


/-- `psumPart` is the product of the symmetric polynomials `psum μᵢ`,
where `μ = (μ₁, μ₂, ...)` is a partition. -/
def psumPart {n : ℕ} (μ : n.Partition) : MvPolynomial σ R := (μ.parts.map (psum σ R)).prod


@[simp]
                                                      /-
                                                        σ : Type u_5
                                                        R : Type u_6
                                                        inst✝¹ : CommSemiring R
                                                        inst✝ : Fintype σ
                                                        ⊢ Eq (MvPolynomial.psum σ R 0) ↑(Fintype.card σ)
                                                      -/
theorem psum_zero : psum σ R 0 = Fintype.card σ := by simp [psum]
                                                      /-
                                                        🎉 no goals
                                                      -/


@[simp]
                                               /-
                                                 σ : Type u_5
                                                 R : Type u_6
                                                 inst✝¹ : CommSemiring R
                                                 inst✝ : Fintype σ
                                                 ⊢ Eq (MvPolynomial.psum σ R 1) (Finset.univ.sum fun i => MvPolynomial.X i)
                                               -/
theorem psum_one : psum σ R 1 = ∑ i, X i := by simp [psum]
                                               /-
                                                 🎉 no goals
                                               -/


@[simp]
                                                               /-
                                                                 σ : Type u_5
                                                                 R : Type u_6
                                                                 inst✝¹ : CommSemiring R
                                                                 inst✝ : Fintype σ
                                                                 ⊢ Eq (MvPolynomial.psumPart σ R (Nat.Partition.indiscrete 0)) 1
                                                               -/
theorem psumPart_zero : psumPart σ R (.indiscrete 0) = 1 := by simp [psumPart]
                                                               /-
                                                                 🎉 no goals
                                                               -/


@[simp]
theorem psumPart_indiscrete {n : ℕ} (npos : n ≠ 0) :
                                                    /-
                                                      σ : Type u_5
                                                      R : Type u_6
                                                      inst✝¹ : CommSemiring R
                                                      inst✝ : Fintype σ
                                                      n : Nat
                                                      npos : Ne n 0
                                                      ⊢ Eq (MvPolynomial.psumPart σ R (Nat.Partition.indiscrete n)) (MvPolynomial.ps …
                                                    -/
    psumPart σ R (.indiscrete n) = psum σ R n := by simp [psumPart, npos]
                                                    /-
                                                      🎉 no goals
                                                    -/


@[simp]
theorem rename_psum (n : ℕ) (e : σ ≃ τ) : rename e (psum σ R n) = psum τ R n := by
  /-
    τ : Type u_2
    σ : Type u_5
    R : Type u_6
    inst✝² : CommSemiring R
    inst✝¹ : Fintype σ
    inst✝ : Fintype τ
    n : Nat
    e : Equiv σ τ
    ⊢ Eq ((MvPolynomial.rename ⇑e) (MvPolynomial.psum σ R n)) (MvPolynomial.psum τ …
  -/
  simp_rw [psum, map_sum, map_pow, rename_X, e.sum_comp (X · ^ n)]
  /-
    🎉 no goals
  -/


theorem psum_isSymmetric (n : ℕ) : IsSymmetric (psum σ R n) := rename_psum _ _ n


/-- The monomial symmetric `MvPolynomial σ R` with exponent set μ.
It is the sum over all the monomials in `MvPolynomial σ R` such that
the multiset of exponents is equal to the multiset of parts of μ. -/
def msymm (μ : n.Partition) : MvPolynomial σ R :=
  ∑ s : {a : Sym σ n // .ofSym a = μ}, (s.1.1.map X).prod


@[simp]
theorem msymm_zero : msymm σ R (.indiscrete 0) = 1 := by
  /-
    σ : Type u_5
    R : Type u_6
    inst✝² : CommSemiring R
    inst✝¹ : Fintype σ
    inst✝ : DecidableEq σ
    ⊢ Eq (MvPolynomial.msymm σ R (Nat.Partition.indiscrete 0)) 1
  -/
  rw [msymm, Fintype.sum_subsingleton _ ⟨(Sym.nil : Sym σ 0), rfl⟩]
  /-
    σ : Type u_5
    R : Type u_6
    inst✝² : CommSemiring R
    inst✝¹ : Fintype σ
    inst✝ : DecidableEq σ
    ⊢ Eq (Multiset.map MvPolynomial.X ↑↑⟨Sym.nil, ⋯⟩).prod 1
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem msymm_one : msymm σ R (.indiscrete 1) = ∑ i, X i := by
  have : (fun (x : Sym σ 1) ↦ x ∈ Set.univ) =
      (fun x ↦ Nat.Partition.ofSym x = Nat.Partition.indiscrete 1) := by
    simp_rw [Set.mem_univ, Nat.Partition.ofSym_one]
  /-
    σ : Type u_5
    R : Type u_6
    inst✝² : CommSemiring R
    inst✝¹ : Fintype σ
    inst✝ : DecidableEq σ
    this : Eq (fun x => Membership.mem Set.univ x) fun x => Eq (Nat.Partition.ofSy …
    ⊢ Eq (MvPolynomial.msymm σ R (Nat.Partition.indiscrete 1)) (Finset.univ.sum fu …
  -/
  symm
  rw [Fintype.sum_equiv (Equiv.trans Sym.oneEquiv (Equiv.Set.univ (Sym σ 1)).symm)
    _ (fun s ↦ (s.1.1.map X).prod)]
    /-
      σ : Type u_5
      R : Type u_6
      inst✝² : CommSemiring R
      inst✝¹ : Fintype σ
      inst✝ : DecidableEq σ
      this : Eq (fun x => Membership.mem Set.univ x) fun x => Eq (Nat.Partition.ofSy …
      ⊢ Eq (Finset.univ.sum fun x => (Multiset.map MvPolynomial.X ↑↑x).prod) (MvPoly …
    -/
  · apply Fintype.sum_equiv (Equiv.subtypeEquivProp this)
    /-
      case h
      σ : Type u_5
      R : Type u_6
      inst✝² : CommSemiring R
      inst✝¹ : Fintype σ
      inst✝ : DecidableEq σ
      this : Eq (fun x => Membership.mem Set.univ x) fun x => Eq (Nat.Partition.ofSy …
      ⊢ ∀ (x : Subtype fun x => Membership.mem Set.univ x), Eq (Multiset.map MvPolyn …
    -/
    intro x
    /-
      case h
      σ : Type u_5
      R : Type u_6
      inst✝² : CommSemiring R
      inst✝¹ : Fintype σ
      inst✝ : DecidableEq σ
      this : Eq (fun x => Membership.mem Set.univ x) fun x => Eq (Nat.Partition.ofSy …
      x : Subtype fun x => Membership.mem Set.univ x
      ⊢ Eq (Multiset.map MvPolynomial.X ↑↑x).prod (Multiset.map MvPolynomial.X ↑↑((E …
    -/
    congr
    /-
      🎉 no goals
    -/
    /-
      σ : Type u_5
      R : Type u_6
      inst✝² : CommSemiring R
      inst✝¹ : Fintype σ
      inst✝ : DecidableEq σ
      this : Eq (fun x => Membership.mem Set.univ x) fun x => Eq (Nat.Partition.ofSy …
      ⊢ ∀ (x : σ), Eq (MvPolynomial.X x) (Multiset.map MvPolynomial.X ↑↑((Sym.oneEqu …
    -/
  · intro x
    /-
      σ : Type u_5
      R : Type u_6
      inst✝² : CommSemiring R
      inst✝¹ : Fintype σ
      inst✝ : DecidableEq σ
      this : Eq (fun x => Membership.mem Set.univ x) fun x => Eq (Nat.Partition.ofSy …
      x : σ
      ⊢ Eq (MvPolynomial.X x) (Multiset.map MvPolynomial.X ↑↑((Sym.oneEquiv.trans (E …
    -/
    rw [← Multiset.prod_singleton (X x), ← Multiset.map_singleton]
    /-
      σ : Type u_5
      R : Type u_6
      inst✝² : CommSemiring R
      inst✝¹ : Fintype σ
      inst✝ : DecidableEq σ
      this : Eq (fun x => Membership.mem Set.univ x) fun x => Eq (Nat.Partition.ofSy …
      x : σ
      ⊢ Eq (Multiset.map MvPolynomial.X (Singleton.singleton x)).prod (Multiset.map  …
    -/
    congr
    /-
      🎉 no goals
    -/


@[simp]
theorem rename_msymm (μ : n.Partition) (e : σ ≃ τ) :
    rename e (msymm σ R μ) = msymm τ R μ := by
  /-
    τ : Type u_2
    σ : Type u_5
    R : Type u_6
    inst✝⁴ : CommSemiring R
    inst✝³ : Fintype σ
    inst✝² : Fintype τ
    inst✝¹ : DecidableEq σ
    inst✝ : DecidableEq τ
    n : Nat
    μ : n.Partition
    e : Equiv σ τ
    ⊢ Eq ((MvPolynomial.rename ⇑e) (MvPolynomial.msymm σ R μ)) (MvPolynomial.msymm …
  -/
  rw [msymm, map_sum]
  /-
    τ : Type u_2
    σ : Type u_5
    R : Type u_6
    inst✝⁴ : CommSemiring R
    inst✝³ : Fintype σ
    inst✝² : Fintype τ
    inst✝¹ : DecidableEq σ
    inst✝ : DecidableEq τ
    n : Nat
    μ : n.Partition
    e : Equiv σ τ
    ⊢ Eq (Finset.univ.sum fun x => (MvPolynomial.rename ⇑e) (Multiset.map MvPolyno …
  -/
  apply Fintype.sum_equiv (Nat.Partition.ofSymShapeEquiv μ e)
  /-
    case h
    τ : Type u_2
    σ : Type u_5
    R : Type u_6
    inst✝⁴ : CommSemiring R
    inst✝³ : Fintype σ
    inst✝² : Fintype τ
    inst✝¹ : DecidableEq σ
    inst✝ : DecidableEq τ
    n : Nat
    μ : n.Partition
    e : Equiv σ τ
    ⊢ ∀ (x : Subtype fun x => Eq (Nat.Partition.ofSym x) μ), Eq ((MvPolynomial.ren …
  -/
  intro
  /-
    case h
    τ : Type u_2
    σ : Type u_5
    R : Type u_6
    inst✝⁴ : CommSemiring R
    inst✝³ : Fintype σ
    inst✝² : Fintype τ
    inst✝¹ : DecidableEq σ
    inst✝ : DecidableEq τ
    n : Nat
    μ : n.Partition
    e : Equiv σ τ
    x✝ : Subtype fun x => Eq (Nat.Partition.ofSym x) μ
    ⊢ Eq ((MvPolynomial.rename ⇑e) (Multiset.map MvPolynomial.X ↑↑x✝).prod) (Multi …
  -/
  rw [← Multiset.prod_hom, Multiset.map_map, Nat.Partition.ofSymShapeEquiv]
  /-
    case h
    τ : Type u_2
    σ : Type u_5
    R : Type u_6
    inst✝⁴ : CommSemiring R
    inst✝³ : Fintype σ
    inst✝² : Fintype τ
    inst✝¹ : DecidableEq σ
    inst✝ : DecidableEq τ
    n : Nat
    μ : n.Partition
    e : Equiv σ τ
    x✝ : Subtype fun x => Eq (Nat.Partition.ofSym x) μ
    ⊢ Eq (Multiset.map (Function.comp (⇑(MvPolynomial.rename ⇑e)) MvPolynomial.X)  …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem msymm_isSymmetric (μ : n.Partition) : IsSymmetric (msymm σ R μ) :=
  rename_msymm _ _ μ


