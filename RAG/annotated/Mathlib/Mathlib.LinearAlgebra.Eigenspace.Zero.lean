lemma IsNilpotent.charpoly_eq_X_pow_finrank {φ : Module.End R M} (h : IsNilpotent φ) :
    φ.charpoly = X ^ finrank R M := by
  /-
    R : Type u_1
    M : Type u_3
    inst✝⁵ : CommRing R
    inst✝⁴ : IsDomain R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : Module.Finite R M
    inst✝ : Module.Free R M
    φ : Module.End R M
    h : IsNilpotent φ
    ⊢ Eq (LinearMap.charpoly φ) (HPow.hPow Polynomial.X (Module.finrank R M))
  -/
  rw [← sub_eq_zero]
  /-
    R : Type u_1
    M : Type u_3
    inst✝⁵ : CommRing R
    inst✝⁴ : IsDomain R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : Module.Finite R M
    inst✝ : Module.Free R M
    φ : Module.End R M
    h : IsNilpotent φ
    ⊢ Eq (HSub.hSub (LinearMap.charpoly φ) (HPow.hPow Polynomial.X (Module.finrank …
  -/
  apply IsNilpotent.eq_zero
  /-
    case h
    R : Type u_1
    M : Type u_3
    inst✝⁵ : CommRing R
    inst✝⁴ : IsDomain R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : Module.Finite R M
    inst✝ : Module.Free R M
    φ : Module.End R M
    h : IsNilpotent φ
    ⊢ IsNilpotent (HSub.hSub (LinearMap.charpoly φ) (HPow.hPow Polynomial.X (Modul …
  -/
  rw [finrank_eq_card_chooseBasisIndex]
  /-
    case h
    R : Type u_1
    M : Type u_3
    inst✝⁵ : CommRing R
    inst✝⁴ : IsDomain R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : Module.Finite R M
    inst✝ : Module.Free R M
    φ : Module.End R M
    h : IsNilpotent φ
    ⊢ IsNilpotent (HSub.hSub (LinearMap.charpoly φ) (HPow.hPow Polynomial.X (Finty …
  -/
  apply Matrix.isNilpotent_charpoly_sub_pow_of_isNilpotent
  /-
    case h.hM
    R : Type u_1
    M : Type u_3
    inst✝⁵ : CommRing R
    inst✝⁴ : IsDomain R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : Module.Finite R M
    inst✝ : Module.Free R M
    φ : Module.End R M
    h : IsNilpotent φ
    ⊢ IsNilpotent ((LinearMap.toMatrix (Module.Free.chooseBasis R M) (Module.Free. …
  -/
  exact h.map (LinearMap.toMatrixAlgEquiv (chooseBasis R M))
  /-
    🎉 no goals
  -/


lemma isNilpotent_iff_charpoly (φ : End R M) :
    IsNilpotent φ ↔ charpoly φ = X ^ finrank R M :=
  ⟨IsNilpotent.charpoly_eq_X_pow_finrank,
                             /-
                               R : Type u_1
                               M : Type u_3
                               inst✝⁵ : CommRing R
                               inst✝⁴ : IsDomain R
                               inst✝³ : AddCommGroup M
                               inst✝² : Module R M
                               inst✝¹ : Module.Finite R M
                               inst✝ : Module.Free R M
                               φ : Module.End R M
                               h : Eq (LinearMap.charpoly φ) (HPow.hPow Polynomial.X (Module.finrank R M))
                               ⊢ Eq (HPow.hPow φ (Module.finrank R M)) 0
                             -/
    fun h ↦ ⟨finrank R M, by rw [← @aeval_X_pow R, ← h, aeval_self_charpoly φ]⟩⟩
                             /-
                               🎉 no goals
                             -/


open Module.Free in
lemma charpoly_nilpotent_tfae [IsNoetherian R M] (φ : Module.End R M) :
    List.TFAE [
      IsNilpotent φ,
      φ.charpoly = X ^ finrank R M,
      ∀ m : M, ∃ (n : ℕ), (φ ^ n) m = 0,
      natTrailingDegree φ.charpoly = finrank R M ] := by
  /-
    R : Type u_1
    M : Type u_3
    inst✝⁶ : CommRing R
    inst✝⁵ : IsDomain R
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : Module.Finite R M
    inst✝¹ : Module.Free R M
    inst✝ : IsNoetherian R M
    φ : Module.End R M
    ⊢ (List.cons (IsNilpotent φ) (List.cons (Eq (LinearMap.charpoly φ) (HPow.hPow  …
  -/
  tfae_have 1 → 2 := IsNilpotent.charpoly_eq_X_pow_finrank
  tfae_have 2 → 3
  | h, m => by
    use finrank R M
    suffices φ ^ finrank R M = 0 by simp only [this, LinearMap.zero_apply]
    simpa only [h, map_pow, aeval_X] using φ.aeval_self_charpoly
  tfae_have 3 → 1
  | h => by
    obtain ⟨n, hn⟩ := Filter.eventually_atTop.mp <| φ.eventually_iSup_ker_pow_eq
    use n
    ext x
    rw [zero_apply, ← mem_ker, ← hn n le_rfl]
    obtain ⟨k, hk⟩ := h x
    rw [← mem_ker] at hk
    exact Submodule.mem_iSup_of_mem _ hk
  tfae_have 2 ↔ 4 := by
    rw [← φ.charpoly_natDegree, φ.charpoly_monic.eq_X_pow_iff_natTrailingDegree_eq_natDegree]
  /-
    R : Type u_1
    M : Type u_3
    inst✝⁶ : CommRing R
    inst✝⁵ : IsDomain R
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : Module.Finite R M
    inst✝¹ : Module.Free R M
    inst✝ : IsNoetherian R M
    φ : Module.End R M
    tfae_1_to_2 : IsNilpotent φ → Eq (LinearMap.charpoly φ) (HPow.hPow Polynomial. …
    tfae_2_to_3 : Eq (LinearMap.charpoly φ) (HPow.hPow Polynomial.X (Module.finran …
    tfae_3_to_1 : (∀ (m : M), Exists fun n => Eq ((HPow.hPow φ n) m) 0) → IsNilpot …
    tfae_2_iff_4 : Iff (Eq (LinearMap.charpoly φ) (HPow.hPow Polynomial.X (Module. …
    ⊢ (List.cons (IsNilpotent φ) (List.cons (Eq (LinearMap.charpoly φ) (HPow.hPow  …
  -/
  tfae_finish
  /-
    🎉 no goals
  -/


lemma charpoly_eq_X_pow_iff [IsNoetherian R M] (φ : Module.End R M) :
    φ.charpoly = X ^ finrank R M ↔ ∀ m : M, ∃ (n : ℕ), (φ ^ n) m = 0 :=
  /-
    R : Type u_1
    M : Type u_3
    inst✝⁶ : CommRing R
    inst✝⁵ : IsDomain R
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : Module.Finite R M
    inst✝¹ : Module.Free R M
    inst✝ : IsNoetherian R M
    φ : Module.End R M
    ⊢ Eq ((List.cons (IsNilpotent φ) (List.cons (Eq (LinearMap.charpoly φ) (HPow.h …
  -/
  /-
    🎉 no goals
  -/
  (charpoly_nilpotent_tfae φ).out 1 2
  /-
    🎉 no goals
  -/


open Module.Free in
lemma hasEigenvalue_zero_tfae (φ : Module.End K M) :
    List.TFAE [
      Module.End.HasEigenvalue φ 0,
      IsRoot (minpoly K φ) 0,
      constantCoeff φ.charpoly = 0,
      LinearMap.det φ = 0,
      ⊥ < ker φ,
      ∃ (m : M), m ≠ 0 ∧ φ m = 0 ] := by
  /-
    K : Type u_2
    M : Type u_3
    inst✝³ : Field K
    inst✝² : AddCommGroup M
    inst✝¹ : Module K M
    inst✝ : Module.Finite K M
    φ : Module.End K M
    ⊢ (List.cons (φ.HasEigenvalue 0) (List.cons ((minpoly K φ).IsRoot 0) (List.con …
  -/
  tfae_have 1 ↔ 2 := Module.End.hasEigenvalue_iff_isRoot
  tfae_have 2 → 3 := by
    obtain ⟨F, hF⟩ := minpoly_dvd_charpoly φ
    simp only [IsRoot.def, constantCoeff_apply, coeff_zero_eq_eval_zero, hF, eval_mul]
    intro h; rw [h, zero_mul]
  tfae_have 3 → 4 := by
    rw [← LinearMap.det_toMatrix (chooseBasis K M), Matrix.det_eq_sign_charpoly_coeff,
      constantCoeff_apply, charpoly]
    intro h; rw [h, mul_zero]
  /-
    K : Type u_2
    M : Type u_3
    inst✝³ : Field K
    inst✝² : AddCommGroup M
    inst✝¹ : Module K M
    inst✝ : Module.Finite K M
    φ : Module.End K M
    tfae_1_iff_2 : Iff (φ.HasEigenvalue 0) ((minpoly K φ).IsRoot 0)
    tfae_2_to_3 : (minpoly K φ).IsRoot 0 → Eq (Polynomial.constantCoeff (LinearMap …
    tfae_3_to_4 : Eq (Polynomial.constantCoeff (LinearMap.charpoly φ)) 0 → Eq (Lin …
    ⊢ (List.cons (φ.HasEigenvalue 0) (List.cons ((minpoly K φ).IsRoot 0) (List.con …
  -/
  tfae_have 4 → 5 := bot_lt_ker_of_det_eq_zero
  tfae_have 5 → 6 := by
    contrapose!
    simp only [not_bot_lt_iff, eq_bot_iff]
    intro h x
    simp only [mem_ker, Submodule.mem_bot]
    contrapose!
    apply h
  tfae_have 6 → 1
  | ⟨x, h1, h2⟩ => by
    apply Module.End.hasEigenvalue_of_hasEigenvector ⟨_, h1⟩
    simpa only [Module.End.eigenspace_zero, mem_ker] using h2
  /-
    K : Type u_2
    M : Type u_3
    inst✝³ : Field K
    inst✝² : AddCommGroup M
    inst✝¹ : Module K M
    inst✝ : Module.Finite K M
    φ : Module.End K M
    tfae_1_iff_2 : Iff (φ.HasEigenvalue 0) ((minpoly K φ).IsRoot 0)
    tfae_2_to_3 : (minpoly K φ).IsRoot 0 → Eq (Polynomial.constantCoeff (LinearMap …
    tfae_3_to_4 : Eq (Polynomial.constantCoeff (LinearMap.charpoly φ)) 0 → Eq (Lin …
    tfae_4_to_5 : Eq (LinearMap.det φ) 0 → LT.lt Bot.bot (LinearMap.ker φ)
    tfae_5_to_6 : LT.lt Bot.bot (LinearMap.ker φ) → Exists fun m => And (Ne m 0) ( …
    tfae_6_to_1 : (Exists fun m => And (Ne m 0) (Eq (φ m) 0)) → φ.HasEigenvalue 0
    ⊢ (List.cons (φ.HasEigenvalue 0) (List.cons ((minpoly K φ).IsRoot 0) (List.con …
  -/
  tfae_finish
  /-
    🎉 no goals
  -/


lemma charpoly_constantCoeff_eq_zero_iff (φ : Module.End K M) :
    constantCoeff φ.charpoly = 0 ↔ ∃ (m : M), m ≠ 0 ∧ φ m = 0 :=
  /-
    K : Type u_2
    M : Type u_3
    inst✝³ : Field K
    inst✝² : AddCommGroup M
    inst✝¹ : Module K M
    inst✝ : Module.Finite K M
    φ : Module.End K M
    ⊢ Eq ((List.cons (φ.HasEigenvalue 0) (List.cons ((minpoly K φ).IsRoot 0) (List …
  -/
  /-
    🎉 no goals
  -/
  (hasEigenvalue_zero_tfae φ).out 2 5
  /-
    🎉 no goals
  -/


open Module.Free in
lemma not_hasEigenvalue_zero_tfae (φ : Module.End K M) :
    List.TFAE [
      ¬ Module.End.HasEigenvalue φ 0,
      ¬ IsRoot (minpoly K φ) 0,
      constantCoeff φ.charpoly ≠ 0,
      LinearMap.det φ ≠ 0,
      ker φ = ⊥,
      ∀ (m : M), φ m = 0 → m = 0 ] := by
  /-
    K : Type u_2
    M : Type u_3
    inst✝³ : Field K
    inst✝² : AddCommGroup M
    inst✝¹ : Module K M
    inst✝ : Module.Finite K M
    φ : Module.End K M
    ⊢ (List.cons (Not (φ.HasEigenvalue 0)) (List.cons (Not ((minpoly K φ).IsRoot 0 …
  -/
  have := (hasEigenvalue_zero_tfae φ).not
  /-
    K : Type u_2
    M : Type u_3
    inst✝³ : Field K
    inst✝² : AddCommGroup M
    inst✝¹ : Module K M
    inst✝ : Module.Finite K M
    φ : Module.End K M
    this : (List.map Not (List.cons (φ.HasEigenvalue 0) (List.cons ((minpoly K φ). …
    ⊢ (List.cons (Not (φ.HasEigenvalue 0)) (List.cons (Not ((minpoly K φ).IsRoot 0 …
  -/
  dsimp only [List.map] at this
  /-
    K : Type u_2
    M : Type u_3
    inst✝³ : Field K
    inst✝² : AddCommGroup M
    inst✝¹ : Module K M
    inst✝ : Module.Finite K M
    φ : Module.End K M
    this : (List.cons (Not (φ.HasEigenvalue 0)) (List.cons (Not ((minpoly K φ).IsR …
    ⊢ (List.cons (Not (φ.HasEigenvalue 0)) (List.cons (Not ((minpoly K φ).IsRoot 0 …
  -/
  push_neg at this
  /-
    K : Type u_2
    M : Type u_3
    inst✝³ : Field K
    inst✝² : AddCommGroup M
    inst✝¹ : Module K M
    inst✝ : Module.Finite K M
    φ : Module.End K M
    this : (List.cons (Not (φ.HasEigenvalue 0)) (List.cons (Not ((minpoly K φ).IsR …
    ⊢ (List.cons (Not (φ.HasEigenvalue 0)) (List.cons (Not ((minpoly K φ).IsRoot 0 …
  -/
  have aux₁ : ∀ m, (m ≠ 0 → φ m ≠ 0) ↔ (φ m = 0 → m = 0) := by intro m; apply not_imp_not
  /-
    K : Type u_2
    M : Type u_3
    inst✝³ : Field K
    inst✝² : AddCommGroup M
    inst✝¹ : Module K M
    inst✝ : Module.Finite K M
    φ : Module.End K M
    this : (List.cons (Not (φ.HasEigenvalue 0)) (List.cons (Not ((minpoly K φ).IsR …
    aux₁ : ∀ (m : M), Iff (Ne m 0 → Ne (φ m) 0) (Eq (φ m) 0 → Eq m 0)
    ⊢ (List.cons (Not (φ.HasEigenvalue 0)) (List.cons (Not ((minpoly K φ).IsRoot 0 …
  -/
  have aux₂ : ker φ = ⊥ ↔ ¬ ⊥ < ker φ := by rw [bot_lt_iff_ne_bot, not_not]
  /-
    K : Type u_2
    M : Type u_3
    inst✝³ : Field K
    inst✝² : AddCommGroup M
    inst✝¹ : Module K M
    inst✝ : Module.Finite K M
    φ : Module.End K M
    this : (List.cons (Not (φ.HasEigenvalue 0)) (List.cons (Not ((minpoly K φ).IsR …
    aux₁ : ∀ (m : M), Iff (Ne m 0 → Ne (φ m) 0) (Eq (φ m) 0 → Eq m 0)
    aux₂ : Iff (Eq (LinearMap.ker φ) Bot.bot) (Not (LT.lt Bot.bot (LinearMap.ker φ …
    ⊢ (List.cons (Not (φ.HasEigenvalue 0)) (List.cons (Not ((minpoly K φ).IsRoot 0 …
  -/
  simpa only [aux₁, aux₂] using this
  /-
    🎉 no goals
  -/


open Module.Free in
lemma finrank_maxGenEigenspace (φ : Module.End K M) :
    finrank K (φ.maxGenEigenspace 0) = natTrailingDegree (φ.charpoly) := by
  /-
    K : Type u_2
    M : Type u_3
    inst✝³ : Field K
    inst✝² : AddCommGroup M
    inst✝¹ : Module K M
    inst✝ : Module.Finite K M
    φ : Module.End K M
    ⊢ Eq (Module.finrank K (Subtype fun x => Membership.mem (φ.maxGenEigenspace 0) …
  -/
  set V := φ.maxGenEigenspace 0
  have hV : V = ⨆ (n : ℕ), ker (φ ^ n) := by
    simp [V, ← Module.End.iSup_genEigenspace_eq, Module.End.genEigenspace_nat]
  /-
    K : Type u_2
    M : Type u_3
    inst✝³ : Field K
    inst✝² : AddCommGroup M
    inst✝¹ : Module K M
    inst✝ : Module.Finite K M
    φ : Module.End K M
    V : Submodule K M := φ.maxGenEigenspace 0
    hV : Eq V (iSup fun n => LinearMap.ker (HPow.hPow φ n))
    ⊢ Eq (Module.finrank K (Subtype fun x => Membership.mem V x)) (LinearMap.charp …
  -/
  let W := ⨅ (n : ℕ), LinearMap.range (φ ^ n)
  have hVW : IsCompl V W := by
    rw [hV]
    exact LinearMap.isCompl_iSup_ker_pow_iInf_range_pow φ
  have hφV : ∀ x ∈ V, φ x ∈ V := by
    simp only [V, Module.End.mem_maxGenEigenspace, zero_smul, sub_zero,
      forall_exists_index]
    intro x n hx
    use n
    rw [← LinearMap.mul_apply, ← pow_succ, pow_succ', LinearMap.mul_apply, hx, map_zero]
  have hφW : ∀ x ∈ W, φ x ∈ W := by
    simp only [W, Submodule.mem_iInf, mem_range]
    intro x H n
    obtain ⟨y, rfl⟩ := H n
    use φ y
    rw [← LinearMap.mul_apply, ← pow_succ, pow_succ', LinearMap.mul_apply]
  /-
    K : Type u_2
    M : Type u_3
    inst✝³ : Field K
    inst✝² : AddCommGroup M
    inst✝¹ : Module K M
    inst✝ : Module.Finite K M
    φ : Module.End K M
    V : Submodule K M := φ.maxGenEigenspace 0
    hV : Eq V (iSup fun n => LinearMap.ker (HPow.hPow φ n))
    W : Submodule K M := iInf fun n => LinearMap.range (HPow.hPow φ n)
    hVW : IsCompl V W
    hφV : ∀ (x : M), Membership.mem V x → Membership.mem V (φ x)
    hφW : ∀ (x : M), Membership.mem W x → Membership.mem W (φ x)
    ⊢ Eq (Module.finrank K (Subtype fun x => Membership.mem V x)) (LinearMap.charp …
  -/
  let F := φ.restrict hφV
  /-
    K : Type u_2
    M : Type u_3
    inst✝³ : Field K
    inst✝² : AddCommGroup M
    inst✝¹ : Module K M
    inst✝ : Module.Finite K M
    φ : Module.End K M
    V : Submodule K M := φ.maxGenEigenspace 0
    hV : Eq V (iSup fun n => LinearMap.ker (HPow.hPow φ n))
    W : Submodule K M := iInf fun n => LinearMap.range (HPow.hPow φ n)
    hVW : IsCompl V W
    hφV : ∀ (x : M), Membership.mem V x → Membership.mem V (φ x)
    hφW : ∀ (x : M), Membership.mem W x → Membership.mem W (φ x)
    F : LinearMap (RingHom.id K) (Subtype fun x => Membership.mem V x) (Subtype fu …
    ⊢ Eq (Module.finrank K (Subtype fun x => Membership.mem V x)) (LinearMap.charp …
  -/
  let G := φ.restrict hφW
  /-
    K : Type u_2
    M : Type u_3
    inst✝³ : Field K
    inst✝² : AddCommGroup M
    inst✝¹ : Module K M
    inst✝ : Module.Finite K M
    φ : Module.End K M
    V : Submodule K M := φ.maxGenEigenspace 0
    hV : Eq V (iSup fun n => LinearMap.ker (HPow.hPow φ n))
    W : Submodule K M := iInf fun n => LinearMap.range (HPow.hPow φ n)
    hVW : IsCompl V W
    hφV : ∀ (x : M), Membership.mem V x → Membership.mem V (φ x)
    hφW : ∀ (x : M), Membership.mem W x → Membership.mem W (φ x)
    F : LinearMap (RingHom.id K) (Subtype fun x => Membership.mem V x) (Subtype fu …
    G : LinearMap (RingHom.id K) (Subtype fun x => Membership.mem W x) (Subtype fu …
    ⊢ Eq (Module.finrank K (Subtype fun x => Membership.mem V x)) (LinearMap.charp …
  -/
  let ψ := F.prodMap G
  /-
    K : Type u_2
    M : Type u_3
    inst✝³ : Field K
    inst✝² : AddCommGroup M
    inst✝¹ : Module K M
    inst✝ : Module.Finite K M
    φ : Module.End K M
    V : Submodule K M := φ.maxGenEigenspace 0
    hV : Eq V (iSup fun n => LinearMap.ker (HPow.hPow φ n))
    W : Submodule K M := iInf fun n => LinearMap.range (HPow.hPow φ n)
    hVW : IsCompl V W
    hφV : ∀ (x : M), Membership.mem V x → Membership.mem V (φ x)
    hφW : ∀ (x : M), Membership.mem W x → Membership.mem W (φ x)
    F : LinearMap (RingHom.id K) (Subtype fun x => Membership.mem V x) (Subtype fu …
    G : LinearMap (RingHom.id K) (Subtype fun x => Membership.mem W x) (Subtype fu …
    ψ : LinearMap (RingHom.id K) (Prod (Subtype fun x => Membership.mem V x) (Subt …
    ⊢ Eq (Module.finrank K (Subtype fun x => Membership.mem V x)) (LinearMap.charp …
  -/
  let e := Submodule.prodEquivOfIsCompl V W hVW
  /-
    K : Type u_2
    M : Type u_3
    inst✝³ : Field K
    inst✝² : AddCommGroup M
    inst✝¹ : Module K M
    inst✝ : Module.Finite K M
    φ : Module.End K M
    V : Submodule K M := φ.maxGenEigenspace 0
    hV : Eq V (iSup fun n => LinearMap.ker (HPow.hPow φ n))
    W : Submodule K M := iInf fun n => LinearMap.range (HPow.hPow φ n)
    hVW : IsCompl V W
    hφV : ∀ (x : M), Membership.mem V x → Membership.mem V (φ x)
    hφW : ∀ (x : M), Membership.mem W x → Membership.mem W (φ x)
    F : LinearMap (RingHom.id K) (Subtype fun x => Membership.mem V x) (Subtype fu …
    G : LinearMap (RingHom.id K) (Subtype fun x => Membership.mem W x) (Subtype fu …
    ψ : LinearMap (RingHom.id K) (Prod (Subtype fun x => Membership.mem V x) (Subt …
    e : LinearEquiv (RingHom.id K) (Prod (Subtype fun x => Membership.mem V x) (Su …
    ⊢ Eq (Module.finrank K (Subtype fun x => Membership.mem V x)) (LinearMap.charp …
  -/
  let bV := chooseBasis K V
  /-
    K : Type u_2
    M : Type u_3
    inst✝³ : Field K
    inst✝² : AddCommGroup M
    inst✝¹ : Module K M
    inst✝ : Module.Finite K M
    φ : Module.End K M
    V : Submodule K M := φ.maxGenEigenspace 0
    hV : Eq V (iSup fun n => LinearMap.ker (HPow.hPow φ n))
    W : Submodule K M := iInf fun n => LinearMap.range (HPow.hPow φ n)
    hVW : IsCompl V W
    hφV : ∀ (x : M), Membership.mem V x → Membership.mem V (φ x)
    hφW : ∀ (x : M), Membership.mem W x → Membership.mem W (φ x)
    F : LinearMap (RingHom.id K) (Subtype fun x => Membership.mem V x) (Subtype fu …
    G : LinearMap (RingHom.id K) (Subtype fun x => Membership.mem W x) (Subtype fu …
    ψ : LinearMap (RingHom.id K) (Prod (Subtype fun x => Membership.mem V x) (Subt …
    e : LinearEquiv (RingHom.id K) (Prod (Subtype fun x => Membership.mem V x) (Su …
    bV : Basis (Module.Free.ChooseBasisIndex K (Subtype fun x => Membership.mem V  …
    ⊢ Eq (Module.finrank K (Subtype fun x => Membership.mem V x)) (LinearMap.charp …
  -/
  let bW := chooseBasis K W
  /-
    K : Type u_2
    M : Type u_3
    inst✝³ : Field K
    inst✝² : AddCommGroup M
    inst✝¹ : Module K M
    inst✝ : Module.Finite K M
    φ : Module.End K M
    V : Submodule K M := φ.maxGenEigenspace 0
    hV : Eq V (iSup fun n => LinearMap.ker (HPow.hPow φ n))
    W : Submodule K M := iInf fun n => LinearMap.range (HPow.hPow φ n)
    hVW : IsCompl V W
    hφV : ∀ (x : M), Membership.mem V x → Membership.mem V (φ x)
    hφW : ∀ (x : M), Membership.mem W x → Membership.mem W (φ x)
    F : LinearMap (RingHom.id K) (Subtype fun x => Membership.mem V x) (Subtype fu …
    G : LinearMap (RingHom.id K) (Subtype fun x => Membership.mem W x) (Subtype fu …
    ψ : LinearMap (RingHom.id K) (Prod (Subtype fun x => Membership.mem V x) (Subt …
    e : LinearEquiv (RingHom.id K) (Prod (Subtype fun x => Membership.mem V x) (Su …
    bV : Basis (Module.Free.ChooseBasisIndex K (Subtype fun x => Membership.mem V  …
    bW : Basis (Module.Free.ChooseBasisIndex K (Subtype fun x => Membership.mem W  …
    ⊢ Eq (Module.finrank K (Subtype fun x => Membership.mem V x)) (LinearMap.charp …
  -/
  let b := bV.prod bW
  have hψ : ψ = e.symm.conj φ := by
    apply b.ext
    simp only [Basis.prod_apply, coe_inl, coe_inr, prodMap_apply, LinearEquiv.conj_apply,
      LinearEquiv.symm_symm, Submodule.coe_prodEquivOfIsCompl, coe_comp, LinearEquiv.coe_coe,
      Function.comp_apply, coprod_apply, Submodule.coe_subtype, map_add, Sum.forall, Sum.elim_inl,
      map_zero, ZeroMemClass.coe_zero, add_zero, LinearEquiv.eq_symm_apply, and_self,
      Submodule.coe_prodEquivOfIsCompl', restrict_coe_apply, implies_true, Sum.elim_inr, zero_add,
      e, V, W, ψ, F, G, b]
  rw [← e.symm.charpoly_conj φ, ← hψ, charpoly_prodMap,
    natTrailingDegree_mul (charpoly_monic _).ne_zero (charpoly_monic _).ne_zero]
  have hG : natTrailingDegree (charpoly G) = 0 := by
    apply Polynomial.natTrailingDegree_eq_zero_of_constantCoeff_ne_zero
    apply ((not_hasEigenvalue_zero_tfae G).out 2 5).mpr
    intro x hx
    apply Subtype.ext
    suffices x.1 ∈ V ⊓ W by rwa [hVW.inf_eq_bot, Submodule.mem_bot] at this
    suffices x.1 ∈ V from ⟨this, x.2⟩
    simp only [Module.End.mem_maxGenEigenspace, zero_smul, sub_zero, V]
    use 1
    rw [pow_one]
    rwa [Subtype.ext_iff] at hx
  /-
    K : Type u_2
    M : Type u_3
    inst✝³ : Field K
    inst✝² : AddCommGroup M
    inst✝¹ : Module K M
    inst✝ : Module.Finite K M
    φ : Module.End K M
    V : Submodule K M := φ.maxGenEigenspace 0
    hV : Eq V (iSup fun n => LinearMap.ker (HPow.hPow φ n))
    W : Submodule K M := iInf fun n => LinearMap.range (HPow.hPow φ n)
    hVW : IsCompl V W
    hφV : ∀ (x : M), Membership.mem V x → Membership.mem V (φ x)
    hφW : ∀ (x : M), Membership.mem W x → Membership.mem W (φ x)
    F : LinearMap (RingHom.id K) (Subtype fun x => Membership.mem V x) (Subtype fu …
    G : LinearMap (RingHom.id K) (Subtype fun x => Membership.mem W x) (Subtype fu …
    ψ : LinearMap (RingHom.id K) (Prod (Subtype fun x => Membership.mem V x) (Subt …
    e : LinearEquiv (RingHom.id K) (Prod (Subtype fun x => Membership.mem V x) (Su …
    bV : Basis (Module.Free.ChooseBasisIndex K (Subtype fun x => Membership.mem V  …
    bW : Basis (Module.Free.ChooseBasisIndex K (Subtype fun x => Membership.mem W  …
    b : Basis (Sum (Module.Free.ChooseBasisIndex K (Subtype fun x => Membership.me …
    hψ : Eq ψ (e.symm.conj φ)
    hG : Eq G.charpoly.natTrailingDegree 0
    ⊢ Eq (Module.finrank K (Subtype fun x => Membership.mem V x)) (HAdd.hAdd F.cha …
  -/
  rw [hG, add_zero, eq_comm]
  /-
    K : Type u_2
    M : Type u_3
    inst✝³ : Field K
    inst✝² : AddCommGroup M
    inst✝¹ : Module K M
    inst✝ : Module.Finite K M
    φ : Module.End K M
    V : Submodule K M := φ.maxGenEigenspace 0
    hV : Eq V (iSup fun n => LinearMap.ker (HPow.hPow φ n))
    W : Submodule K M := iInf fun n => LinearMap.range (HPow.hPow φ n)
    hVW : IsCompl V W
    hφV : ∀ (x : M), Membership.mem V x → Membership.mem V (φ x)
    hφW : ∀ (x : M), Membership.mem W x → Membership.mem W (φ x)
    F : LinearMap (RingHom.id K) (Subtype fun x => Membership.mem V x) (Subtype fu …
    G : LinearMap (RingHom.id K) (Subtype fun x => Membership.mem W x) (Subtype fu …
    ψ : LinearMap (RingHom.id K) (Prod (Subtype fun x => Membership.mem V x) (Subt …
    e : LinearEquiv (RingHom.id K) (Prod (Subtype fun x => Membership.mem V x) (Su …
    bV : Basis (Module.Free.ChooseBasisIndex K (Subtype fun x => Membership.mem V  …
    bW : Basis (Module.Free.ChooseBasisIndex K (Subtype fun x => Membership.mem W  …
    b : Basis (Sum (Module.Free.ChooseBasisIndex K (Subtype fun x => Membership.me …
    hψ : Eq ψ (e.symm.conj φ)
    hG : Eq G.charpoly.natTrailingDegree 0
    ⊢ Eq F.charpoly.natTrailingDegree (Module.finrank K (Subtype fun x => Membersh …
  -/
  apply ((charpoly_nilpotent_tfae F).out 2 3).mp
  /-
    K : Type u_2
    M : Type u_3
    inst✝³ : Field K
    inst✝² : AddCommGroup M
    inst✝¹ : Module K M
    inst✝ : Module.Finite K M
    φ : Module.End K M
    V : Submodule K M := φ.maxGenEigenspace 0
    hV : Eq V (iSup fun n => LinearMap.ker (HPow.hPow φ n))
    W : Submodule K M := iInf fun n => LinearMap.range (HPow.hPow φ n)
    hVW : IsCompl V W
    hφV : ∀ (x : M), Membership.mem V x → Membership.mem V (φ x)
    hφW : ∀ (x : M), Membership.mem W x → Membership.mem W (φ x)
    F : LinearMap (RingHom.id K) (Subtype fun x => Membership.mem V x) (Subtype fu …
    G : LinearMap (RingHom.id K) (Subtype fun x => Membership.mem W x) (Subtype fu …
    ψ : LinearMap (RingHom.id K) (Prod (Subtype fun x => Membership.mem V x) (Subt …
    e : LinearEquiv (RingHom.id K) (Prod (Subtype fun x => Membership.mem V x) (Su …
    bV : Basis (Module.Free.ChooseBasisIndex K (Subtype fun x => Membership.mem V  …
    bW : Basis (Module.Free.ChooseBasisIndex K (Subtype fun x => Membership.mem W  …
    b : Basis (Sum (Module.Free.ChooseBasisIndex K (Subtype fun x => Membership.me …
    hψ : Eq ψ (e.symm.conj φ)
    hG : Eq G.charpoly.natTrailingDegree 0
    ⊢ ∀ (m : Subtype fun x => Membership.mem V x), Exists fun n => Eq ((HPow.hPow  …
  -/
  simp only [Subtype.forall, Module.End.mem_maxGenEigenspace, zero_smul, sub_zero, V, F]
  /-
    K : Type u_2
    M : Type u_3
    inst✝³ : Field K
    inst✝² : AddCommGroup M
    inst✝¹ : Module K M
    inst✝ : Module.Finite K M
    φ : Module.End K M
    V : Submodule K M := φ.maxGenEigenspace 0
    hV : Eq V (iSup fun n => LinearMap.ker (HPow.hPow φ n))
    W : Submodule K M := iInf fun n => LinearMap.range (HPow.hPow φ n)
    hVW : IsCompl V W
    hφV : ∀ (x : M), Membership.mem V x → Membership.mem V (φ x)
    hφW : ∀ (x : M), Membership.mem W x → Membership.mem W (φ x)
    F : LinearMap (RingHom.id K) (Subtype fun x => Membership.mem V x) (Subtype fu …
    G : LinearMap (RingHom.id K) (Subtype fun x => Membership.mem W x) (Subtype fu …
    ψ : LinearMap (RingHom.id K) (Prod (Subtype fun x => Membership.mem V x) (Subt …
    e : LinearEquiv (RingHom.id K) (Prod (Subtype fun x => Membership.mem V x) (Su …
    bV : Basis (Module.Free.ChooseBasisIndex K (Subtype fun x => Membership.mem V  …
    bW : Basis (Module.Free.ChooseBasisIndex K (Subtype fun x => Membership.mem W  …
    b : Basis (Sum (Module.Free.ChooseBasisIndex K (Subtype fun x => Membership.me …
    hψ : Eq ψ (e.symm.conj φ)
    hG : Eq G.charpoly.natTrailingDegree 0
    ⊢ ∀ (a : M) (b : Exists fun k => Eq ((HPow.hPow φ k) a) 0), Exists fun n => Eq …
  -/
  rintro x ⟨n, hx⟩
  /-
    case intro
    K : Type u_2
    M : Type u_3
    inst✝³ : Field K
    inst✝² : AddCommGroup M
    inst✝¹ : Module K M
    inst✝ : Module.Finite K M
    φ : Module.End K M
    V : Submodule K M := φ.maxGenEigenspace 0
    hV : Eq V (iSup fun n => LinearMap.ker (HPow.hPow φ n))
    W : Submodule K M := iInf fun n => LinearMap.range (HPow.hPow φ n)
    hVW : IsCompl V W
    hφV : ∀ (x : M), Membership.mem V x → Membership.mem V (φ x)
    hφW : ∀ (x : M), Membership.mem W x → Membership.mem W (φ x)
    F : LinearMap (RingHom.id K) (Subtype fun x => Membership.mem V x) (Subtype fu …
    G : LinearMap (RingHom.id K) (Subtype fun x => Membership.mem W x) (Subtype fu …
    ψ : LinearMap (RingHom.id K) (Prod (Subtype fun x => Membership.mem V x) (Subt …
    e : LinearEquiv (RingHom.id K) (Prod (Subtype fun x => Membership.mem V x) (Su …
    bV : Basis (Module.Free.ChooseBasisIndex K (Subtype fun x => Membership.mem V  …
    bW : Basis (Module.Free.ChooseBasisIndex K (Subtype fun x => Membership.mem W  …
    b : Basis (Sum (Module.Free.ChooseBasisIndex K (Subtype fun x => Membership.me …
    hψ : Eq ψ (e.symm.conj φ)
    hG : Eq G.charpoly.natTrailingDegree 0
    x : M
    n : Nat
    hx : Eq ((HPow.hPow φ n) x) 0
    ⊢ Exists fun n_1 => Eq ((HPow.hPow (LinearMap.restrict φ hφV) n_1) ⟨x, ⋯⟩) 0
  -/
  use n
  /-
    case h
    K : Type u_2
    M : Type u_3
    inst✝³ : Field K
    inst✝² : AddCommGroup M
    inst✝¹ : Module K M
    inst✝ : Module.Finite K M
    φ : Module.End K M
    V : Submodule K M := φ.maxGenEigenspace 0
    hV : Eq V (iSup fun n => LinearMap.ker (HPow.hPow φ n))
    W : Submodule K M := iInf fun n => LinearMap.range (HPow.hPow φ n)
    hVW : IsCompl V W
    hφV : ∀ (x : M), Membership.mem V x → Membership.mem V (φ x)
    hφW : ∀ (x : M), Membership.mem W x → Membership.mem W (φ x)
    F : LinearMap (RingHom.id K) (Subtype fun x => Membership.mem V x) (Subtype fu …
    G : LinearMap (RingHom.id K) (Subtype fun x => Membership.mem W x) (Subtype fu …
    ψ : LinearMap (RingHom.id K) (Prod (Subtype fun x => Membership.mem V x) (Subt …
    e : LinearEquiv (RingHom.id K) (Prod (Subtype fun x => Membership.mem V x) (Su …
    bV : Basis (Module.Free.ChooseBasisIndex K (Subtype fun x => Membership.mem V  …
    bW : Basis (Module.Free.ChooseBasisIndex K (Subtype fun x => Membership.mem W  …
    b : Basis (Sum (Module.Free.ChooseBasisIndex K (Subtype fun x => Membership.me …
    hψ : Eq ψ (e.symm.conj φ)
    hG : Eq G.charpoly.natTrailingDegree 0
    x : M
    n : Nat
    hx : Eq ((HPow.hPow φ n) x) 0
    ⊢ Eq ((HPow.hPow (LinearMap.restrict φ hφV) n) ⟨x, ⋯⟩) 0
  -/
  apply Subtype.ext
  /-
    case h.a
    K : Type u_2
    M : Type u_3
    inst✝³ : Field K
    inst✝² : AddCommGroup M
    inst✝¹ : Module K M
    inst✝ : Module.Finite K M
    φ : Module.End K M
    V : Submodule K M := φ.maxGenEigenspace 0
    hV : Eq V (iSup fun n => LinearMap.ker (HPow.hPow φ n))
    W : Submodule K M := iInf fun n => LinearMap.range (HPow.hPow φ n)
    hVW : IsCompl V W
    hφV : ∀ (x : M), Membership.mem V x → Membership.mem V (φ x)
    hφW : ∀ (x : M), Membership.mem W x → Membership.mem W (φ x)
    F : LinearMap (RingHom.id K) (Subtype fun x => Membership.mem V x) (Subtype fu …
    G : LinearMap (RingHom.id K) (Subtype fun x => Membership.mem W x) (Subtype fu …
    ψ : LinearMap (RingHom.id K) (Prod (Subtype fun x => Membership.mem V x) (Subt …
    e : LinearEquiv (RingHom.id K) (Prod (Subtype fun x => Membership.mem V x) (Su …
    bV : Basis (Module.Free.ChooseBasisIndex K (Subtype fun x => Membership.mem V  …
    bW : Basis (Module.Free.ChooseBasisIndex K (Subtype fun x => Membership.mem W  …
    b : Basis (Sum (Module.Free.ChooseBasisIndex K (Subtype fun x => Membership.me …
    hψ : Eq ψ (e.symm.conj φ)
    hG : Eq G.charpoly.natTrailingDegree 0
    x : M
    n : Nat
    hx : Eq ((HPow.hPow φ n) x) 0
    ⊢ Eq ↑((HPow.hPow (LinearMap.restrict φ hφV) n) ⟨x, ⋯⟩) ↑0
  -/
  rw [ZeroMemClass.coe_zero]
  /-
    case h.a
    K : Type u_2
    M : Type u_3
    inst✝³ : Field K
    inst✝² : AddCommGroup M
    inst✝¹ : Module K M
    inst✝ : Module.Finite K M
    φ : Module.End K M
    V : Submodule K M := φ.maxGenEigenspace 0
    hV : Eq V (iSup fun n => LinearMap.ker (HPow.hPow φ n))
    W : Submodule K M := iInf fun n => LinearMap.range (HPow.hPow φ n)
    hVW : IsCompl V W
    hφV : ∀ (x : M), Membership.mem V x → Membership.mem V (φ x)
    hφW : ∀ (x : M), Membership.mem W x → Membership.mem W (φ x)
    F : LinearMap (RingHom.id K) (Subtype fun x => Membership.mem V x) (Subtype fu …
    G : LinearMap (RingHom.id K) (Subtype fun x => Membership.mem W x) (Subtype fu …
    ψ : LinearMap (RingHom.id K) (Prod (Subtype fun x => Membership.mem V x) (Subt …
    e : LinearEquiv (RingHom.id K) (Prod (Subtype fun x => Membership.mem V x) (Su …
    bV : Basis (Module.Free.ChooseBasisIndex K (Subtype fun x => Membership.mem V  …
    bW : Basis (Module.Free.ChooseBasisIndex K (Subtype fun x => Membership.mem W  …
    b : Basis (Sum (Module.Free.ChooseBasisIndex K (Subtype fun x => Membership.me …
    hψ : Eq ψ (e.symm.conj φ)
    hG : Eq G.charpoly.natTrailingDegree 0
    x : M
    n : Nat
    hx : Eq ((HPow.hPow φ n) x) 0
    ⊢ Eq (↑((HPow.hPow (LinearMap.restrict φ hφV) n) ⟨x, ⋯⟩)) 0
  -/
  refine .trans ?_ hx
  /-
    case h.a
    K : Type u_2
    M : Type u_3
    inst✝³ : Field K
    inst✝² : AddCommGroup M
    inst✝¹ : Module K M
    inst✝ : Module.Finite K M
    φ : Module.End K M
    V : Submodule K M := φ.maxGenEigenspace 0
    hV : Eq V (iSup fun n => LinearMap.ker (HPow.hPow φ n))
    W : Submodule K M := iInf fun n => LinearMap.range (HPow.hPow φ n)
    hVW : IsCompl V W
    hφV : ∀ (x : M), Membership.mem V x → Membership.mem V (φ x)
    hφW : ∀ (x : M), Membership.mem W x → Membership.mem W (φ x)
    F : LinearMap (RingHom.id K) (Subtype fun x => Membership.mem V x) (Subtype fu …
    G : LinearMap (RingHom.id K) (Subtype fun x => Membership.mem W x) (Subtype fu …
    ψ : LinearMap (RingHom.id K) (Prod (Subtype fun x => Membership.mem V x) (Subt …
    e : LinearEquiv (RingHom.id K) (Prod (Subtype fun x => Membership.mem V x) (Su …
    bV : Basis (Module.Free.ChooseBasisIndex K (Subtype fun x => Membership.mem V  …
    bW : Basis (Module.Free.ChooseBasisIndex K (Subtype fun x => Membership.mem W  …
    b : Basis (Sum (Module.Free.ChooseBasisIndex K (Subtype fun x => Membership.me …
    hψ : Eq ψ (e.symm.conj φ)
    hG : Eq G.charpoly.natTrailingDegree 0
    x : M
    n : Nat
    hx : Eq ((HPow.hPow φ n) x) 0
    ⊢ Eq (↑((HPow.hPow (LinearMap.restrict φ hφV) n) ⟨x, ⋯⟩)) ((HPow.hPow φ n) x)
  -/
  generalize_proofs h'
  /-
    case h.a
    K : Type u_2
    M : Type u_3
    inst✝³ : Field K
    inst✝² : AddCommGroup M
    inst✝¹ : Module K M
    inst✝ : Module.Finite K M
    φ : Module.End K M
    V : Submodule K M := φ.maxGenEigenspace 0
    hV : Eq V (iSup fun n => LinearMap.ker (HPow.hPow φ n))
    W : Submodule K M := iInf fun n => LinearMap.range (HPow.hPow φ n)
    hVW : IsCompl V W
    hφV : ∀ (x : M), Membership.mem V x → Membership.mem V (φ x)
    hφW : ∀ (x : M), Membership.mem W x → Membership.mem W (φ x)
    F : LinearMap (RingHom.id K) (Subtype fun x => Membership.mem V x) (Subtype fu …
    G : LinearMap (RingHom.id K) (Subtype fun x => Membership.mem W x) (Subtype fu …
    ψ : LinearMap (RingHom.id K) (Prod (Subtype fun x => Membership.mem V x) (Subt …
    e : LinearEquiv (RingHom.id K) (Prod (Subtype fun x => Membership.mem V x) (Su …
    bV : Basis (Module.Free.ChooseBasisIndex K (Subtype fun x => Membership.mem V  …
    bW : Basis (Module.Free.ChooseBasisIndex K (Subtype fun x => Membership.mem W  …
    b : Basis (Sum (Module.Free.ChooseBasisIndex K (Subtype fun x => Membership.me …
    hψ : Eq ψ (e.symm.conj φ)
    hG : Eq G.charpoly.natTrailingDegree 0
    x : M
    n : Nat
    hx : Eq ((HPow.hPow φ n) x) 0
    h' : Membership.mem (φ.maxGenEigenspace 0) x
    ⊢ Eq (↑((HPow.hPow (LinearMap.restrict φ hφV) n) ⟨x, h'⟩)) ((HPow.hPow φ n) x)
  -/
  clear hx
  induction n with
  | zero => simp only [pow_zero, one_apply]
  | succ n ih => simp only [pow_succ', LinearMap.mul_apply, ih, restrict_apply]


