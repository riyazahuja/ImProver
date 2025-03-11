theorem mem_iInf_maxGenEigenspace_iff (χ : ι → R) (m : M) :
    m ∈ ⨅ i, (f i).maxGenEigenspace (χ i) ↔ ∀ j, ∃ k : ℕ, ((f j - χ j • ↑1) ^ k) m = 0 := by
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_4
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    f : ι → Module.End R M
    χ : ι → R
    m : M
    ⊢ Iff (Membership.mem (iInf fun i => (f i).maxGenEigenspace (χ i)) m) (∀ (j :  …
  -/
  simp
  /-
    🎉 no goals
  -/


/-- Given a family of endomorphisms `i ↦ f i`, a family of candidate eigenvalues `i ↦ μ i`, and a
submodule `p` which is invariant wrt every `f i`, the intersection of `p` with the simultaneous
maximal generalised eigenspace (taken over all `i`), is the same as the simultaneous maximal
generalised eigenspace of the `f i` restricted to `p`. -/
lemma _root_.Submodule.inf_iInf_maxGenEigenspace_of_forall_mapsTo {μ : ι → R}
    (p : Submodule R M) (hfp : ∀ i, MapsTo (f i) p p) :
    p ⊓ ⨅ i, (f i).maxGenEigenspace (μ i) =
      (⨅ i, maxGenEigenspace ((f i).restrict (hfp i)) (μ i)).map p.subtype := by
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_4
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    f : ι → Module.End R M
    μ : ι → R
    p : Submodule R M
    hfp : ∀ (i : ι), Set.MapsTo ⇑(f i) ↑p ↑p
    ⊢ Eq (Min.min p (iInf fun i => (f i).maxGenEigenspace (μ i))) (Submodule.map p …
  -/
  cases isEmpty_or_nonempty ι
    /-
      case inl
      ι : Type u_1
      R : Type u_2
      M : Type u_4
      inst✝² : CommRing R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      f : ι → Module.End R M
      μ : ι → R
      p : Submodule R M
      hfp : ∀ (i : ι), Set.MapsTo ⇑(f i) ↑p ↑p
      h✝ : IsEmpty ι
      ⊢ Eq (Min.min p (iInf fun i => (f i).maxGenEigenspace (μ i))) (Submodule.map p …
    -/
  · simp [iInf_of_isEmpty]
    /-
      🎉 no goals
    -/
    /-
      case inr
      ι : Type u_1
      R : Type u_2
      M : Type u_4
      inst✝² : CommRing R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      f : ι → Module.End R M
      μ : ι → R
      p : Submodule R M
      hfp : ∀ (i : ι), Set.MapsTo ⇑(f i) ↑p ↑p
      h✝ : Nonempty ι
      ⊢ Eq (Min.min p (iInf fun i => (f i).maxGenEigenspace (μ i))) (Submodule.map p …
    -/
  · simp_rw [inf_iInf, p.inf_genEigenspace _ (hfp _), Submodule.map_iInf _ p.injective_subtype]
    /-
      🎉 no goals
    -/


/-- Given a family of endomorphisms `i ↦ f i`, a family of candidate eigenvalues `i ↦ μ i`, and a
distinguished index `i` whose maximal generalised `μ i`-eigenspace is invariant wrt every `f j`,
taking simultaneous maximal generalised eigenspaces is unaffected by first restricting to the
distinguished generalised `μ i`-eigenspace. -/
lemma iInf_maxGenEigenspace_restrict_map_subtype_eq
    {μ : ι → R} (i : ι)
    (h : ∀ j, MapsTo (f j) ((f i).maxGenEigenspace (μ i)) ((f i).maxGenEigenspace (μ i))) :
    letI p := (f i).maxGenEigenspace (μ i)
    letI q (j : ι) := maxGenEigenspace ((f j).restrict (h j)) (μ j)
    (⨅ j, q j).map p.subtype = ⨅ j, (f j).maxGenEigenspace (μ j) := by
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_4
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    f : ι → Module.End R M
    μ : ι → R
    i : ι
    h : ∀ (j : ι), Set.MapsTo ⇑(f j) ↑((f i).maxGenEigenspace (μ i)) ↑((f i).maxGe …
    ⊢ Eq (Submodule.map ((f i).maxGenEigenspace (μ i)).subtype (iInf fun j => (fun …
  -/
  have : Nonempty ι := ⟨i⟩
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_4
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    f : ι → Module.End R M
    μ : ι → R
    i : ι
    h : ∀ (j : ι), Set.MapsTo ⇑(f j) ↑((f i).maxGenEigenspace (μ i)) ↑((f i).maxGe …
    this : Nonempty ι
    ⊢ Eq (Submodule.map ((f i).maxGenEigenspace (μ i)).subtype (iInf fun j => (fun …
  -/
  set p := (f i).maxGenEigenspace (μ i)
  have : ⨅ j, (f j).maxGenEigenspace (μ j) = p ⊓ ⨅ j, (f j).maxGenEigenspace (μ j) := by
    refine le_antisymm ?_ inf_le_right
    simpa only [le_inf_iff, le_refl, and_true] using iInf_le _ _
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_4
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    f : ι → Module.End R M
    μ : ι → R
    i : ι
    h✝ : ∀ (j : ι), Set.MapsTo ⇑(f j) ↑((f i).maxGenEigenspace (μ i)) ↑((f i).maxG …
    this✝ : Nonempty ι
    p : Submodule R M := (f i).maxGenEigenspace (μ i)
    h : ∀ (j : ι), Set.MapsTo ⇑(f j) ↑p ↑p
    this : Eq (iInf fun j => (f j).maxGenEigenspace (μ j)) (Min.min p (iInf fun j  …
    ⊢ Eq (Submodule.map ((f i).maxGenEigenspace (μ i)).subtype (iInf fun j => (fun …
  -/
  rw [Submodule.map_iInf _ p.injective_subtype, this, Submodule.inf_iInf]
  conv_rhs =>
    enter [1]
    ext
    rw [p.inf_genEigenspace (f _) (h _)]


lemma disjoint_iInf_maxGenEigenspace {χ₁ χ₂ : ι → R} (h : χ₁ ≠ χ₂) :
    Disjoint (⨅ i, (f i).maxGenEigenspace (χ₁ i)) (⨅ i, (f i).maxGenEigenspace (χ₂ i)) := by
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_4
    inst✝³ : CommRing R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    f : ι → Module.End R M
    inst✝ : NoZeroSMulDivisors R M
    χ₁ χ₂ : ι → R
    h : Ne χ₁ χ₂
    ⊢ Disjoint (iInf fun i => (f i).maxGenEigenspace (χ₁ i)) (iInf fun i => (f i). …
  -/
  obtain ⟨j, hj⟩ : ∃ j, χ₁ j ≠ χ₂ j := Function.ne_iff.mp h
  /-
    case intro
    ι : Type u_1
    R : Type u_2
    M : Type u_4
    inst✝³ : CommRing R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    f : ι → Module.End R M
    inst✝ : NoZeroSMulDivisors R M
    χ₁ χ₂ : ι → R
    h : Ne χ₁ χ₂
    j : ι
    hj : Ne (χ₁ j) (χ₂ j)
    ⊢ Disjoint (iInf fun i => (f i).maxGenEigenspace (χ₁ i)) (iInf fun i => (f i). …
  -/
  exact (End.disjoint_genEigenspace (f j) hj ⊤ ⊤).mono (iInf_le _ j) (iInf_le _ j)
  /-
    🎉 no goals
  -/


lemma injOn_iInf_maxGenEigenspace :
    InjOn (fun χ : ι → R ↦ ⨅ i, (f i).maxGenEigenspace (χ i))
      {χ | ⨅ i, (f i).maxGenEigenspace (χ i) ≠ ⊥} := by
  rintro χ₁ _ χ₂
    hχ₂ (hχ₁₂ : ⨅ i, (f i).maxGenEigenspace (χ₁ i) = ⨅ i, (f i).maxGenEigenspace (χ₂ i))
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_4
    inst✝³ : CommRing R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    f : ι → Module.End R M
    inst✝ : NoZeroSMulDivisors R M
    χ₁ : ι → R
    a✝ : Membership.mem (setOf fun χ => Ne (iInf fun i => (f i).maxGenEigenspace ( …
    χ₂ : ι → R
    hχ₂ : Membership.mem (setOf fun χ => Ne (iInf fun i => (f i).maxGenEigenspace  …
    hχ₁₂ : Eq (iInf fun i => (f i).maxGenEigenspace (χ₁ i)) (iInf fun i => (f i).m …
    ⊢ Eq χ₁ χ₂
  -/
  contrapose! hχ₂
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_4
    inst✝³ : CommRing R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    f : ι → Module.End R M
    inst✝ : NoZeroSMulDivisors R M
    χ₁ : ι → R
    a✝ : Membership.mem (setOf fun χ => Ne (iInf fun i => (f i).maxGenEigenspace ( …
    χ₂ : ι → R
    hχ₁₂ : Eq (iInf fun i => (f i).maxGenEigenspace (χ₁ i)) (iInf fun i => (f i).m …
    hχ₂ : Ne χ₁ χ₂
    ⊢ Not (Membership.mem (setOf fun χ => Ne (iInf fun i => (f i).maxGenEigenspace …
  -/
  simpa [hχ₁₂] using disjoint_iInf_maxGenEigenspace f hχ₂
  /-
    🎉 no goals
  -/


lemma independent_iInf_maxGenEigenspace_of_forall_mapsTo
    (h : ∀ i j φ, MapsTo (f i) ((f j).maxGenEigenspace φ) ((f j).maxGenEigenspace φ)) :
    iSupIndep fun χ : ι → R ↦ ⨅ i, (f i).maxGenEigenspace (χ i) := by
  replace h (l : ι) (χ : ι → R) :
      MapsTo (f l) (⨅ i, (f i).maxGenEigenspace (χ i)) (⨅ i, (f i).maxGenEigenspace (χ i)) := by
    intro x hx
    simp only [iInf_eq_iInter, mem_iInter, SetLike.mem_coe] at hx ⊢
    exact fun i ↦ h l i (χ i) (hx i)
  classical
  suffices ∀ χ (s : Finset (ι → R)) (_ : χ ∉ s),
      Disjoint (⨅ i, (f i).maxGenEigenspace (χ i))
        (s.sup fun (χ : ι → R) ↦ ⨅ i, (f i).maxGenEigenspace (χ i)) by
    simpa only [iSupIndep_iff_supIndep_of_injOn (injOn_iInf_maxGenEigenspace f),
      Finset.supIndep_iff_disjoint_erase] using fun s χ _ ↦ this _ _ (s.not_mem_erase χ)
  intro χ₁ s
  induction s using Finset.induction_on with
  | empty => simp
  | insert _n ih =>
  rename_i χ₂ s
  intro hχ₁₂
  obtain ⟨hχ₁₂ : χ₁ ≠ χ₂, hχ₁ : χ₁ ∉ s⟩ := by rwa [Finset.mem_insert, not_or] at hχ₁₂
  specialize ih hχ₁
  rw [Finset.sup_insert, disjoint_iff, Submodule.eq_bot_iff]
  rintro x ⟨hx, hx'⟩
  simp only [SetLike.mem_coe] at hx hx'
  suffices x ∈ ⨅ i, (f i).maxGenEigenspace (χ₂ i) by
    rw [← Submodule.mem_bot (R := R), ← (disjoint_iInf_maxGenEigenspace f hχ₁₂).eq_bot]
    exact ⟨hx, this⟩
  obtain ⟨y, hy, z, hz, rfl⟩ := Submodule.mem_sup.mp hx'; clear hx'
  suffices ∀ l, ∃ (k : ℕ),
      ((f l - algebraMap R (Module.End R M) (χ₂ l)) ^ k) (y + z) ∈
      (⨅ i, (f i).maxGenEigenspace (χ₁ i)) ⊓
        Finset.sup s fun χ ↦ ⨅ i, (f i).maxGenEigenspace (χ i) by
    simpa [ih.eq_bot, Submodule.mem_bot] using this
  intro l
  let g : Module.End R M := f l - algebraMap R (Module.End R M) (χ₂ l)
  obtain ⟨k, hk : (g ^ k) y = 0⟩ := (mem_iInf_maxGenEigenspace_iff _ _ _).mp hy l
  have aux (f : End R M) (φ : R) (k : ℕ) (p : Submodule R M) (hp : MapsTo f p p) :
      MapsTo ((f - algebraMap R (Module.End R M) φ) ^ k) p p := by
    rw [LinearMap.coe_pow]
    exact MapsTo.iterate (fun m hm ↦ p.sub_mem (hp hm) (p.smul_mem _ hm)) k
  refine ⟨k, Submodule.mem_inf.mp ⟨?_, ?_⟩⟩
  · refine aux (f l) (χ₂ l) k (⨅ i, (f i).maxGenEigenspace (χ₁ i)) ?_ hx
    simp only [Submodule.iInf_coe]
    exact h l χ₁
  · rw [map_add, hk, zero_add]
    suffices (s.sup fun χ ↦ (⨅ i, (f i).maxGenEigenspace (χ i))).map (g ^ k) ≤
        s.sup fun χ ↦ (⨅ i, (f i).maxGenEigenspace (χ i)) by
      refine this (Submodule.mem_map_of_mem ?_)
      simp_rw [Finset.sup_eq_iSup, ← Finset.sup_eq_iSup] at hz
      exact hz
    simp_rw [Finset.sup_eq_iSup, Submodule.map_iSup (ι := ι → R), Submodule.map_iSup (ι := _ ∈ s)]
    refine iSup₂_mono fun χ _ ↦ ?_
    rintro - ⟨u, hu, rfl⟩
    refine aux (f l) (χ₂ l) k (⨅ i, (f i).maxGenEigenspace (χ i)) ?_ hu
    simp only [Submodule.iInf_coe]
    exact h l χ


/-- Given a family of endomorphisms `i ↦ f i` which are compatible in the sense that every maximal
generalised eigenspace of `f i` is invariant wrt `f j`, if each `f i` is triangularizable, the
family is simultaneously triangularizable. -/
lemma iSup_iInf_maxGenEigenspace_eq_top_of_forall_mapsTo [FiniteDimensional K M]
    (f : ι → End K M)
    (h : ∀ i j φ, MapsTo (f i) ((f j).maxGenEigenspace φ) ((f j).maxGenEigenspace φ))
    (h' : ∀ i, ⨆ μ, (f i).maxGenEigenspace μ = ⊤) :
    ⨆ χ : ι → K, ⨅ i, (f i).maxGenEigenspace (χ i) = ⊤ := by
  /-
    ι : Type u_1
    K : Type u_3
    M : Type u_4
    inst✝³ : Field K
    inst✝² : AddCommGroup M
    inst✝¹ : Module K M
    inst✝ : FiniteDimensional K M
    f : ι → Module.End K M
    h : ∀ (i j : ι) (φ : K), Set.MapsTo ⇑(f i) ↑((f j).maxGenEigenspace φ) ↑((f j) …
    h' : ∀ (i : ι), Eq (iSup fun μ => (f i).maxGenEigenspace μ) Top.top
    ⊢ Eq (iSup fun χ => iInf fun i => (f i).maxGenEigenspace (χ i)) Top.top
  -/
  generalize h_dim : finrank K M = n
  /-
    ι : Type u_1
    K : Type u_3
    M : Type u_4
    inst✝³ : Field K
    inst✝² : AddCommGroup M
    inst✝¹ : Module K M
    inst✝ : FiniteDimensional K M
    f : ι → Module.End K M
    h : ∀ (i j : ι) (φ : K), Set.MapsTo ⇑(f i) ↑((f j).maxGenEigenspace φ) ↑((f j) …
    h' : ∀ (i : ι), Eq (iSup fun μ => (f i).maxGenEigenspace μ) Top.top
    n : Nat
    h_dim : Eq (Module.finrank K M) n
    ⊢ Eq (iSup fun χ => iInf fun i => (f i).maxGenEigenspace (χ i)) Top.top
  -/
  induction n using Nat.strongRecOn generalizing M with | ind n ih => ?_
  obtain this | ⟨i : ι, hy : ¬ ∃ φ, (f i).maxGenEigenspace φ = ⊤⟩ :=
    forall_or_exists_not (fun j : ι ↦ ∃ φ : K, (f j).maxGenEigenspace φ = ⊤)
    /-
      case ind.inl
      ι : Type u_1
      K : Type u_3
      inst✝³ : Field K
      n : Nat
      ih : ∀ (m : Nat), LT.lt m n → ∀ {M : Type u_4} [inst : AddCommGroup M] [inst_1 …
      M : Type u_4
      inst✝² : AddCommGroup M
      inst✝¹ : Module K M
      inst✝ : FiniteDimensional K M
      f : ι → Module.End K M
      h : ∀ (i j : ι) (φ : K), Set.MapsTo ⇑(f i) ↑((f j).maxGenEigenspace φ) ↑((f j) …
      h' : ∀ (i : ι), Eq (iSup fun μ => (f i).maxGenEigenspace μ) Top.top
      h_dim : Eq (Module.finrank K M) n
      this : ∀ (a : ι), Exists fun φ => Eq ((f a).maxGenEigenspace φ) Top.top
      ⊢ Eq (iSup fun χ => iInf fun i => (f i).maxGenEigenspace (χ i)) Top.top
    -/
  · choose χ hχ using this
    /-
      case ind.inl
      ι : Type u_1
      K : Type u_3
      inst✝³ : Field K
      n : Nat
      ih : ∀ (m : Nat), LT.lt m n → ∀ {M : Type u_4} [inst : AddCommGroup M] [inst_1 …
      M : Type u_4
      inst✝² : AddCommGroup M
      inst✝¹ : Module K M
      inst✝ : FiniteDimensional K M
      f : ι → Module.End K M
      h : ∀ (i j : ι) (φ : K), Set.MapsTo ⇑(f i) ↑((f j).maxGenEigenspace φ) ↑((f j) …
      h' : ∀ (i : ι), Eq (iSup fun μ => (f i).maxGenEigenspace μ) Top.top
      h_dim : Eq (Module.finrank K M) n
      χ : ι → K
      hχ : ∀ (a : ι), Eq ((f a).maxGenEigenspace (χ a)) Top.top
      ⊢ Eq (iSup fun χ => iInf fun i => (f i).maxGenEigenspace (χ i)) Top.top
    -/
    replace hχ : ⨅ i, (f i).maxGenEigenspace (χ i) = ⊤ := by simpa
    /-
      case ind.inl
      ι : Type u_1
      K : Type u_3
      inst✝³ : Field K
      n : Nat
      ih : ∀ (m : Nat), LT.lt m n → ∀ {M : Type u_4} [inst : AddCommGroup M] [inst_1 …
      M : Type u_4
      inst✝² : AddCommGroup M
      inst✝¹ : Module K M
      inst✝ : FiniteDimensional K M
      f : ι → Module.End K M
      h : ∀ (i j : ι) (φ : K), Set.MapsTo ⇑(f i) ↑((f j).maxGenEigenspace φ) ↑((f j) …
      h' : ∀ (i : ι), Eq (iSup fun μ => (f i).maxGenEigenspace μ) Top.top
      h_dim : Eq (Module.finrank K M) n
      χ : ι → K
      hχ : Eq (iInf fun i => (f i).maxGenEigenspace (χ i)) Top.top
      ⊢ Eq (iSup fun χ => iInf fun i => (f i).maxGenEigenspace (χ i)) Top.top
    -/
    simp_rw [eq_top_iff] at hχ ⊢
    /-
      case ind.inl
      ι : Type u_1
      K : Type u_3
      inst✝³ : Field K
      n : Nat
      ih : ∀ (m : Nat), LT.lt m n → ∀ {M : Type u_4} [inst : AddCommGroup M] [inst_1 …
      M : Type u_4
      inst✝² : AddCommGroup M
      inst✝¹ : Module K M
      inst✝ : FiniteDimensional K M
      f : ι → Module.End K M
      h : ∀ (i j : ι) (φ : K), Set.MapsTo ⇑(f i) ↑((f j).maxGenEigenspace φ) ↑((f j) …
      h' : ∀ (i : ι), Eq (iSup fun μ => (f i).maxGenEigenspace μ) Top.top
      h_dim : Eq (Module.finrank K M) n
      χ : ι → K
      hχ : LE.le Top.top (iInf fun i => (f i).maxGenEigenspace (χ i))
      ⊢ LE.le Top.top (iSup fun χ => iInf fun i => (f i).maxGenEigenspace (χ i))
    -/
    exact le_trans hχ <| le_iSup (fun χ : ι → K ↦ ⨅ i, (f i).maxGenEigenspace (χ i)) χ
    /-
      🎉 no goals
    -/
  · replace hy : ∀ φ, finrank K ((f i).maxGenEigenspace φ) < n := fun φ ↦ by
      simp_rw [not_exists, ← lt_top_iff_ne_top] at hy; exact h_dim ▸ Submodule.finrank_lt (hy φ)
    have hi (j : ι) (φ : K) :
        MapsTo (f j) ((f i).maxGenEigenspace φ) ((f i).maxGenEigenspace φ) := by
      exact h j i φ
    replace ih (φ : K) :
        ⨆ χ : ι → K, ⨅ j, maxGenEigenspace ((f j).restrict (hi j φ)) (χ j) = ⊤ := by
      apply ih _ (hy φ)
      · intro j k μ
        exact mapsTo_restrict_maxGenEigenspace_restrict_of_mapsTo (f j) (f k) _ _ (h j k μ)
      · exact fun j ↦ Module.End.genEigenspace_restrict_eq_top _ (h' j)
      · rfl
    replace ih (φ : K) :
        ⨆ (χ : ι → K) (_ : χ i = φ), ⨅ j, maxGenEigenspace ((f j).restrict (hi j φ)) (χ j) = ⊤ := by
      suffices ∀ χ : ι → K, χ i ≠ φ → ⨅ j, maxGenEigenspace ((f j).restrict (hi j φ)) (χ j) = ⊥ by
        specialize ih φ; rw [iSup_split, biSup_congr this] at ih; simpa using ih
      intro χ hχ
      rw [eq_bot_iff, ← ((f i).maxGenEigenspace φ).ker_subtype, LinearMap.ker,
        ← Submodule.map_le_iff_le_comap, ← Submodule.inf_iInf_maxGenEigenspace_of_forall_mapsTo,
        ← disjoint_iff_inf_le]
      exact ((f i).disjoint_genEigenspace hχ.symm _ _).mono_right (iInf_le _ i)
    replace ih (φ : K) :
        ⨆ (χ : ι → K) (_ : χ i = φ), ⨅ j, maxGenEigenspace (f j) (χ j) =
        maxGenEigenspace (f i) φ := by
      have (χ : ι → K) (hχ : χ i = φ) : ⨅ j, maxGenEigenspace (f j) (χ j) =
          (⨅ j, maxGenEigenspace ((f j).restrict (hi j φ)) (χ j)).map
            ((f i).maxGenEigenspace φ).subtype := by
        rw [← hχ, iInf_maxGenEigenspace_restrict_map_subtype_eq]
      simp_rw [biSup_congr this, ← Submodule.map_iSup, ih, Submodule.map_top,
        Submodule.range_subtype]
    /-
      case ind.inr.intro
      ι : Type u_1
      K : Type u_3
      inst✝³ : Field K
      n : Nat
      M : Type u_4
      inst✝² : AddCommGroup M
      inst✝¹ : Module K M
      inst✝ : FiniteDimensional K M
      f : ι → Module.End K M
      h : ∀ (i j : ι) (φ : K), Set.MapsTo ⇑(f i) ↑((f j).maxGenEigenspace φ) ↑((f j) …
      h' : ∀ (i : ι), Eq (iSup fun μ => (f i).maxGenEigenspace μ) Top.top
      h_dim : Eq (Module.finrank K M) n
      i : ι
      hy : ∀ (φ : K), LT.lt (Module.finrank K (Subtype fun x => Membership.mem ((f i …
      hi : ∀ (j : ι) (φ : K), Set.MapsTo ⇑(f j) ↑((f i).maxGenEigenspace φ) ↑((f i). …
      ih : ∀ (φ : K), Eq (iSup fun χ => iSup fun x => iInf fun j => (f j).maxGenEige …
      ⊢ Eq (iSup fun χ => iInf fun i => (f i).maxGenEigenspace (χ i)) Top.top
    -/
    simpa only [← ih, iSup_comm (ι := K), iSup_iSup_eq_right] using h' i
    /-
      🎉 no goals
    -/


/-- A commuting family of triangularizable endomorphisms is simultaneously triangularizable. -/
theorem iSup_iInf_maxGenEigenspace_eq_top_of_iSup_maxGenEigenspace_eq_top_of_commute
    [FiniteDimensional K M] (f : ι → Module.End K M) (h : Pairwise fun i j ↦ Commute (f i) (f j))
    (h' : ∀ i, ⨆ μ, (f i).maxGenEigenspace μ = ⊤) :
    ⨆ χ : ι → K, ⨅ i, (f i).maxGenEigenspace (χ i) = ⊤ := by
  refine Module.End.iSup_iInf_maxGenEigenspace_eq_top_of_forall_mapsTo _
    (fun i j ↦ Module.End.mapsTo_maxGenEigenspace_of_comm ?_) h'
  /-
    ι : Type u_1
    K : Type u_3
    M : Type u_4
    inst✝³ : Field K
    inst✝² : AddCommGroup M
    inst✝¹ : Module K M
    inst✝ : FiniteDimensional K M
    f : ι → Module.End K M
    h : Pairwise fun i j => Commute (f i) (f j)
    h' : ∀ (i : ι), Eq (iSup fun μ => (f i).maxGenEigenspace μ) Top.top
    i j : ι
    ⊢ Commute (f j) (f i)
  -/
                                         /-
                                           🎉 no goals
                                         -/
  rcases eq_or_ne j i with rfl | hij <;> tauto
                                         /-
                                           🎉 no goals
                                         -/


