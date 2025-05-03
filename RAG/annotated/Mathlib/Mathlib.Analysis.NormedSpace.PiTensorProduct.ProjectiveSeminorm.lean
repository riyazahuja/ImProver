/-- A lift of the projective seminorm to `FreeAddMonoid (𝕜 × Π i, Eᵢ)`, useful to prove the
properties of `projectiveSeminorm`.
-/
def projectiveSeminormAux : FreeAddMonoid (𝕜 × Π i, E i) → ℝ :=
  List.sum ∘ (List.map (fun p ↦ ‖p.1‖ * ∏ i, ‖p.2 i‖))


theorem projectiveSeminormAux_nonneg (p : FreeAddMonoid (𝕜 × Π i, E i)) :
    0 ≤ projectiveSeminormAux p := by
  /-
    ι : Type uι
    inst✝² : Fintype ι
    𝕜 : Type u𝕜
    inst✝¹ : NontriviallyNormedField 𝕜
    E : ι → Type uE
    inst✝ : (i : ι) → SeminormedAddCommGroup (E i)
    p : FreeAddMonoid (Prod 𝕜 ((i : ι) → E i))
    ⊢ LE.le 0 (PiTensorProduct.projectiveSeminormAux p)
  -/
  simp only [projectiveSeminormAux, Function.comp_apply]
  /-
    ι : Type uι
    inst✝² : Fintype ι
    𝕜 : Type u𝕜
    inst✝¹ : NontriviallyNormedField 𝕜
    E : ι → Type uE
    inst✝ : (i : ι) → SeminormedAddCommGroup (E i)
    p : FreeAddMonoid (Prod 𝕜 ((i : ι) → E i))
    ⊢ LE.le 0 (List.map (fun p => HMul.hMul (Norm.norm p.1) (Finset.univ.prod fun  …
  -/
  refine List.sum_nonneg ?_
  /-
    ι : Type uι
    inst✝² : Fintype ι
    𝕜 : Type u𝕜
    inst✝¹ : NontriviallyNormedField 𝕜
    E : ι → Type uE
    inst✝ : (i : ι) → SeminormedAddCommGroup (E i)
    p : FreeAddMonoid (Prod 𝕜 ((i : ι) → E i))
    ⊢ ∀ (x : Real), Membership.mem (List.map (fun p => HMul.hMul (Norm.norm p.1) ( …
  -/
  intro a
  simp only [Multiset.map_coe, Multiset.mem_coe, List.mem_map, Prod.exists, forall_exists_index,
    and_imp]
  /-
    ι : Type uι
    inst✝² : Fintype ι
    𝕜 : Type u𝕜
    inst✝¹ : NontriviallyNormedField 𝕜
    E : ι → Type uE
    inst✝ : (i : ι) → SeminormedAddCommGroup (E i)
    p : FreeAddMonoid (Prod 𝕜 ((i : ι) → E i))
    a : Real
    ⊢ ∀ (x : 𝕜) (x_1 : (i : ι) → E i), Membership.mem p { fst := x, snd := x_1 } → …
  -/
  intro x m _ h
  /-
    ι : Type uι
    inst✝² : Fintype ι
    𝕜 : Type u𝕜
    inst✝¹ : NontriviallyNormedField 𝕜
    E : ι → Type uE
    inst✝ : (i : ι) → SeminormedAddCommGroup (E i)
    p : FreeAddMonoid (Prod 𝕜 ((i : ι) → E i))
    a : Real
    x : 𝕜
    m : (i : ι) → E i
    a✝ : Membership.mem p { fst := x, snd := m }
    h : Eq (HMul.hMul (Norm.norm x) (Finset.univ.prod fun x => Norm.norm (m x))) a
    ⊢ LE.le 0 a
  -/
  rw [← h]
  /-
    ι : Type uι
    inst✝² : Fintype ι
    𝕜 : Type u𝕜
    inst✝¹ : NontriviallyNormedField 𝕜
    E : ι → Type uE
    inst✝ : (i : ι) → SeminormedAddCommGroup (E i)
    p : FreeAddMonoid (Prod 𝕜 ((i : ι) → E i))
    a : Real
    x : 𝕜
    m : (i : ι) → E i
    a✝ : Membership.mem p { fst := x, snd := m }
    h : Eq (HMul.hMul (Norm.norm x) (Finset.univ.prod fun x => Norm.norm (m x))) a
    ⊢ LE.le 0 (HMul.hMul (Norm.norm x) (Finset.univ.prod fun x => Norm.norm (m x)))
  -/
  exact mul_nonneg (norm_nonneg _) (Finset.prod_nonneg (fun _ _ ↦ norm_nonneg _))
  /-
    🎉 no goals
  -/


theorem projectiveSeminormAux_add_le (p q : FreeAddMonoid (𝕜 × Π i, E i)) :
    projectiveSeminormAux (p + q) ≤ projectiveSeminormAux p + projectiveSeminormAux q := by
  /-
    ι : Type uι
    inst✝² : Fintype ι
    𝕜 : Type u𝕜
    inst✝¹ : NontriviallyNormedField 𝕜
    E : ι → Type uE
    inst✝ : (i : ι) → SeminormedAddCommGroup (E i)
    p q : FreeAddMonoid (Prod 𝕜 ((i : ι) → E i))
    ⊢ LE.le (PiTensorProduct.projectiveSeminormAux (HAdd.hAdd p q)) (HAdd.hAdd (Pi …
  -/
  simp only [projectiveSeminormAux, Function.comp_apply, Multiset.map_coe, Multiset.sum_coe]
  /-
    ι : Type uι
    inst✝² : Fintype ι
    𝕜 : Type u𝕜
    inst✝¹ : NontriviallyNormedField 𝕜
    E : ι → Type uE
    inst✝ : (i : ι) → SeminormedAddCommGroup (E i)
    p q : FreeAddMonoid (Prod 𝕜 ((i : ι) → E i))
    ⊢ LE.le (List.map (fun p => HMul.hMul (Norm.norm p.1) (Finset.univ.prod fun i  …
  -/
  erw [List.map_append]
  /-
    ι : Type uι
    inst✝² : Fintype ι
    𝕜 : Type u𝕜
    inst✝¹ : NontriviallyNormedField 𝕜
    E : ι → Type uE
    inst✝ : (i : ι) → SeminormedAddCommGroup (E i)
    p q : FreeAddMonoid (Prod 𝕜 ((i : ι) → E i))
    ⊢ LE.le (HAppend.hAppend (List.map (fun p => HMul.hMul (Norm.norm p.1) (Finset …
  -/
  rw [List.sum_append]
  /-
    ι : Type uι
    inst✝² : Fintype ι
    𝕜 : Type u𝕜
    inst✝¹ : NontriviallyNormedField 𝕜
    E : ι → Type uE
    inst✝ : (i : ι) → SeminormedAddCommGroup (E i)
    p q : FreeAddMonoid (Prod 𝕜 ((i : ι) → E i))
    ⊢ LE.le (HAdd.hAdd (List.map (fun p => HMul.hMul (Norm.norm p.1) (Finset.univ. …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem projectiveSeminormAux_smul (p : FreeAddMonoid (𝕜 × Π i, E i)) (a : 𝕜) :
    projectiveSeminormAux (List.map (fun (y : 𝕜 × Π i, E i) ↦ (a * y.1, y.2)) p) =
    ‖a‖ * projectiveSeminormAux p := by
  simp only [projectiveSeminormAux, Function.comp_apply, Multiset.map_coe, List.map_map,
    Multiset.sum_coe]
  /-
    ι : Type uι
    inst✝² : Fintype ι
    𝕜 : Type u𝕜
    inst✝¹ : NontriviallyNormedField 𝕜
    E : ι → Type uE
    inst✝ : (i : ι) → SeminormedAddCommGroup (E i)
    p : FreeAddMonoid (Prod 𝕜 ((i : ι) → E i))
    a : 𝕜
    ⊢ Eq (List.map (Function.comp (fun p => HMul.hMul (Norm.norm p.1) (Finset.univ …
  -/
  rw [← smul_eq_mul, List.smul_sum, ← List.comp_map]
  /-
    ι : Type uι
    inst✝² : Fintype ι
    𝕜 : Type u𝕜
    inst✝¹ : NontriviallyNormedField 𝕜
    E : ι → Type uE
    inst✝ : (i : ι) → SeminormedAddCommGroup (E i)
    p : FreeAddMonoid (Prod 𝕜 ((i : ι) → E i))
    a : 𝕜
    ⊢ Eq (List.map (Function.comp (fun p => HMul.hMul (Norm.norm p.1) (Finset.univ …
  -/
  congr 2
  /-
    case e_a.e_f
    ι : Type uι
    inst✝² : Fintype ι
    𝕜 : Type u𝕜
    inst✝¹ : NontriviallyNormedField 𝕜
    E : ι → Type uE
    inst✝ : (i : ι) → SeminormedAddCommGroup (E i)
    p : FreeAddMonoid (Prod 𝕜 ((i : ι) → E i))
    a : 𝕜
    ⊢ Eq (Function.comp (fun p => HMul.hMul (Norm.norm p.1) (Finset.univ.prod fun  …
  -/
  ext x
  /-
    case e_a.e_f.h
    ι : Type uι
    inst✝² : Fintype ι
    𝕜 : Type u𝕜
    inst✝¹ : NontriviallyNormedField 𝕜
    E : ι → Type uE
    inst✝ : (i : ι) → SeminormedAddCommGroup (E i)
    p : FreeAddMonoid (Prod 𝕜 ((i : ι) → E i))
    a : 𝕜
    x : Prod 𝕜 ((i : ι) → E i)
    ⊢ Eq (Function.comp (fun p => HMul.hMul (Norm.norm p.1) (Finset.univ.prod fun  …
  -/
  simp only [Function.comp_apply, norm_mul, smul_eq_mul]
  /-
    case e_a.e_f.h
    ι : Type uι
    inst✝² : Fintype ι
    𝕜 : Type u𝕜
    inst✝¹ : NontriviallyNormedField 𝕜
    E : ι → Type uE
    inst✝ : (i : ι) → SeminormedAddCommGroup (E i)
    p : FreeAddMonoid (Prod 𝕜 ((i : ι) → E i))
    a : 𝕜
    x : Prod 𝕜 ((i : ι) → E i)
    ⊢ Eq (HMul.hMul (HMul.hMul (Norm.norm a) (Norm.norm x.1)) (Finset.univ.prod fu …
  -/
  rw [mul_assoc]
  /-
    🎉 no goals
  -/


theorem bddBelow_projectiveSemiNormAux (x : ⨂[𝕜] i, E i) :
    BddBelow (Set.range (fun (p : lifts x) ↦ projectiveSeminormAux p.1)) := by
  /-
    ι : Type uι
    inst✝³ : Fintype ι
    𝕜 : Type u𝕜
    inst✝² : NontriviallyNormedField 𝕜
    E : ι → Type uE
    inst✝¹ : (i : ι) → SeminormedAddCommGroup (E i)
    inst✝ : (i : ι) → NormedSpace 𝕜 (E i)
    x : PiTensorProduct 𝕜 fun i => E i
    ⊢ BddBelow (Set.range fun p => PiTensorProduct.projectiveSeminormAux ↑p)
  -/
  existsi 0
  /-
    ι : Type uι
    inst✝³ : Fintype ι
    𝕜 : Type u𝕜
    inst✝² : NontriviallyNormedField 𝕜
    E : ι → Type uE
    inst✝¹ : (i : ι) → SeminormedAddCommGroup (E i)
    inst✝ : (i : ι) → NormedSpace 𝕜 (E i)
    x : PiTensorProduct 𝕜 fun i => E i
    ⊢ Membership.mem (lowerBounds (Set.range fun p => PiTensorProduct.projectiveSe …
  -/
  rw [mem_lowerBounds]
  simp only [Set.mem_range, Subtype.exists, exists_prop, forall_exists_index, and_imp,
    forall_apply_eq_imp_iff₂]
  /-
    ι : Type uι
    inst✝³ : Fintype ι
    𝕜 : Type u𝕜
    inst✝² : NontriviallyNormedField 𝕜
    E : ι → Type uE
    inst✝¹ : (i : ι) → SeminormedAddCommGroup (E i)
    inst✝ : (i : ι) → NormedSpace 𝕜 (E i)
    x : PiTensorProduct 𝕜 fun i => E i
    ⊢ ∀ (a : FreeAddMonoid (Prod 𝕜 ((i : ι) → E i))), Membership.mem x.lifts a → L …
  -/
  exact fun p _ ↦ projectiveSeminormAux_nonneg p
  /-
    🎉 no goals
  -/


/-- The projective seminorm on `⨂[𝕜] i, Eᵢ`. It sends an element `x` of `⨂[𝕜] i, Eᵢ` to the
infimum over all expressions of `x` as `∑ j, ⨂ₜ[𝕜] mⱼ i` (with the `mⱼ` ∈ `Π i, Eᵢ`)
of `∑ j, Π i, ‖mⱼ i‖`.
-/
noncomputable def projectiveSeminorm : Seminorm 𝕜 (⨂[𝕜] i, E i) := by
  /-
    ι : Type uι
    inst✝³ : Fintype ι
    𝕜 : Type u𝕜
    inst✝² : NontriviallyNormedField 𝕜
    E : ι → Type uE
    inst✝¹ : (i : ι) → SeminormedAddCommGroup (E i)
    inst✝ : (i : ι) → NormedSpace 𝕜 (E i)
    ⊢ Seminorm 𝕜 (PiTensorProduct 𝕜 fun i => E i)
  -/
  refine Seminorm.ofSMulLE (fun x ↦ iInf (fun (p : lifts x) ↦ projectiveSeminormAux p.1)) ?_ ?_ ?_
    /-
      case refine_1
      ι : Type uι
      inst✝³ : Fintype ι
      𝕜 : Type u𝕜
      inst✝² : NontriviallyNormedField 𝕜
      E : ι → Type uE
      inst✝¹ : (i : ι) → SeminormedAddCommGroup (E i)
      inst✝ : (i : ι) → NormedSpace 𝕜 (E i)
      ⊢ Eq ((fun x => iInf fun p => PiTensorProduct.projectiveSeminormAux ↑p) 0) 0
    -/
  · refine le_antisymm ?_ ?_
      /-
        case refine_1.refine_1
        ι : Type uι
        inst✝³ : Fintype ι
        𝕜 : Type u𝕜
        inst✝² : NontriviallyNormedField 𝕜
        E : ι → Type uE
        inst✝¹ : (i : ι) → SeminormedAddCommGroup (E i)
        inst✝ : (i : ι) → NormedSpace 𝕜 (E i)
        ⊢ LE.le ((fun x => iInf fun p => PiTensorProduct.projectiveSeminormAux ↑p) 0) 0
      -/
    · refine ciInf_le_of_le (bddBelow_projectiveSemiNormAux (0 : ⨂[𝕜] i, E i)) ⟨0, lifts_zero⟩ ?_
      /-
        case refine_1.refine_1
        ι : Type uι
        inst✝³ : Fintype ι
        𝕜 : Type u𝕜
        inst✝² : NontriviallyNormedField 𝕜
        E : ι → Type uE
        inst✝¹ : (i : ι) → SeminormedAddCommGroup (E i)
        inst✝ : (i : ι) → NormedSpace 𝕜 (E i)
        ⊢ LE.le (PiTensorProduct.projectiveSeminormAux ↑⟨0, ⋯⟩) 0
      -/
      simp only [projectiveSeminormAux, Function.comp_apply]
      /-
        case refine_1.refine_1
        ι : Type uι
        inst✝³ : Fintype ι
        𝕜 : Type u𝕜
        inst✝² : NontriviallyNormedField 𝕜
        E : ι → Type uE
        inst✝¹ : (i : ι) → SeminormedAddCommGroup (E i)
        inst✝ : (i : ι) → NormedSpace 𝕜 (E i)
        ⊢ LE.le (List.map (fun p => HMul.hMul (Norm.norm p.1) (Finset.univ.prod fun i  …
      -/
      rw [List.sum_eq_zero]
      /-
        case refine_1.refine_1
        ι : Type uι
        inst✝³ : Fintype ι
        𝕜 : Type u𝕜
        inst✝² : NontriviallyNormedField 𝕜
        E : ι → Type uE
        inst✝¹ : (i : ι) → SeminormedAddCommGroup (E i)
        inst✝ : (i : ι) → NormedSpace 𝕜 (E i)
        ⊢ ∀ (x : Real), Membership.mem (List.map (fun p => HMul.hMul (Norm.norm p.1) ( …
      -/
      intro _
      /-
        case refine_1.refine_1
        ι : Type uι
        inst✝³ : Fintype ι
        𝕜 : Type u𝕜
        inst✝² : NontriviallyNormedField 𝕜
        E : ι → Type uE
        inst✝¹ : (i : ι) → SeminormedAddCommGroup (E i)
        inst✝ : (i : ι) → NormedSpace 𝕜 (E i)
        x✝ : Real
        ⊢ Membership.mem (List.map (fun p => HMul.hMul (Norm.norm p.1) (Finset.univ.pr …
      -/
      simp only [List.mem_map, Prod.exists, forall_exists_index, and_imp]
      /-
        case refine_1.refine_1
        ι : Type uι
        inst✝³ : Fintype ι
        𝕜 : Type u𝕜
        inst✝² : NontriviallyNormedField 𝕜
        E : ι → Type uE
        inst✝¹ : (i : ι) → SeminormedAddCommGroup (E i)
        inst✝ : (i : ι) → NormedSpace 𝕜 (E i)
        x✝ : Real
        ⊢ ∀ (x : 𝕜) (x_1 : (i : ι) → E i), Membership.mem 0 { fst := x, snd := x_1 } → …
      -/
      intro _ _ hxm
      /-
        case refine_1.refine_1
        ι : Type uι
        inst✝³ : Fintype ι
        𝕜 : Type u𝕜
        inst✝² : NontriviallyNormedField 𝕜
        E : ι → Type uE
        inst✝¹ : (i : ι) → SeminormedAddCommGroup (E i)
        inst✝ : (i : ι) → NormedSpace 𝕜 (E i)
        x✝² : Real
        x✝¹ : 𝕜
        x✝ : (i : ι) → E i
        hxm : Membership.mem 0 { fst := x✝¹, snd := x✝ }
        ⊢ Eq (HMul.hMul (Norm.norm x✝¹) (Finset.univ.prod fun x => Norm.norm (x✝ x)))  …
      -/
      rw [← FreeAddMonoid.ofList_nil] at hxm
      /-
        case refine_1.refine_1
        ι : Type uι
        inst✝³ : Fintype ι
        𝕜 : Type u𝕜
        inst✝² : NontriviallyNormedField 𝕜
        E : ι → Type uE
        inst✝¹ : (i : ι) → SeminormedAddCommGroup (E i)
        inst✝ : (i : ι) → NormedSpace 𝕜 (E i)
        x✝² : Real
        x✝¹ : 𝕜
        x✝ : (i : ι) → E i
        hxm : Membership.mem (FreeAddMonoid.ofList List.nil) { fst := x✝¹, snd := x✝ }
        ⊢ Eq (HMul.hMul (Norm.norm x✝¹) (Finset.univ.prod fun x => Norm.norm (x✝ x)))  …
      -/
      exfalso
      /-
        case refine_1.refine_1
        ι : Type uι
        inst✝³ : Fintype ι
        𝕜 : Type u𝕜
        inst✝² : NontriviallyNormedField 𝕜
        E : ι → Type uE
        inst✝¹ : (i : ι) → SeminormedAddCommGroup (E i)
        inst✝ : (i : ι) → NormedSpace 𝕜 (E i)
        x✝² : Real
        x✝¹ : 𝕜
        x✝ : (i : ι) → E i
        hxm : Membership.mem (FreeAddMonoid.ofList List.nil) { fst := x✝¹, snd := x✝ }
        ⊢ False
      -/
      exact List.not_mem_nil _ hxm
      /-
        🎉 no goals
      -/
      /-
        case refine_1.refine_2
        ι : Type uι
        inst✝³ : Fintype ι
        𝕜 : Type u𝕜
        inst✝² : NontriviallyNormedField 𝕜
        E : ι → Type uE
        inst✝¹ : (i : ι) → SeminormedAddCommGroup (E i)
        inst✝ : (i : ι) → NormedSpace 𝕜 (E i)
        ⊢ LE.le 0 ((fun x => iInf fun p => PiTensorProduct.projectiveSeminormAux ↑p) 0)
      -/
    · letI : Nonempty (lifts 0) := ⟨0, lifts_zero (R := 𝕜) (s := E)⟩
      /-
        case refine_1.refine_2
        ι : Type uι
        inst✝³ : Fintype ι
        𝕜 : Type u𝕜
        inst✝² : NontriviallyNormedField 𝕜
        E : ι → Type uE
        inst✝¹ : (i : ι) → SeminormedAddCommGroup (E i)
        inst✝ : (i : ι) → NormedSpace 𝕜 (E i)
        this : Nonempty ↑(PiTensorProduct.lifts 0) := Nonempty.intro ⟨0, PiTensorProdu …
        ⊢ LE.le 0 ((fun x => iInf fun p => PiTensorProduct.projectiveSeminormAux ↑p) 0)
      -/
      exact le_ciInf (fun p ↦ projectiveSeminormAux_nonneg p.1)
      /-
        🎉 no goals
      -/
    /-
      case refine_2
      ι : Type uι
      inst✝³ : Fintype ι
      𝕜 : Type u𝕜
      inst✝² : NontriviallyNormedField 𝕜
      E : ι → Type uE
      inst✝¹ : (i : ι) → SeminormedAddCommGroup (E i)
      inst✝ : (i : ι) → NormedSpace 𝕜 (E i)
      ⊢ ∀ (x y : PiTensorProduct 𝕜 fun i => E i), LE.le ((fun x => iInf fun p => PiT …
    -/
  · intro x y
    /-
      case refine_2
      ι : Type uι
      inst✝³ : Fintype ι
      𝕜 : Type u𝕜
      inst✝² : NontriviallyNormedField 𝕜
      E : ι → Type uE
      inst✝¹ : (i : ι) → SeminormedAddCommGroup (E i)
      inst✝ : (i : ι) → NormedSpace 𝕜 (E i)
      x y : PiTensorProduct 𝕜 fun i => E i
      ⊢ LE.le ((fun x => iInf fun p => PiTensorProduct.projectiveSeminormAux ↑p) (HA …
    -/
    letI := nonempty_subtype.mpr (nonempty_lifts x); letI := nonempty_subtype.mpr (nonempty_lifts y)
    exact le_ciInf_add_ciInf (fun p q ↦ ciInf_le_of_le (bddBelow_projectiveSemiNormAux _)
      ⟨p.1 + q.1, lifts_add p.2 q.2⟩ (projectiveSeminormAux_add_le p.1 q.1))
    /-
      case refine_3
      ι : Type uι
      inst✝³ : Fintype ι
      𝕜 : Type u𝕜
      inst✝² : NontriviallyNormedField 𝕜
      E : ι → Type uE
      inst✝¹ : (i : ι) → SeminormedAddCommGroup (E i)
      inst✝ : (i : ι) → NormedSpace 𝕜 (E i)
      ⊢ ∀ (r : 𝕜) (x : PiTensorProduct 𝕜 fun i => E i), LE.le ((fun x => iInf fun p  …
    -/
  · intro a x
    /-
      case refine_3
      ι : Type uι
      inst✝³ : Fintype ι
      𝕜 : Type u𝕜
      inst✝² : NontriviallyNormedField 𝕜
      E : ι → Type uE
      inst✝¹ : (i : ι) → SeminormedAddCommGroup (E i)
      inst✝ : (i : ι) → NormedSpace 𝕜 (E i)
      a : 𝕜
      x : PiTensorProduct 𝕜 fun i => E i
      ⊢ LE.le ((fun x => iInf fun p => PiTensorProduct.projectiveSeminormAux ↑p) (HS …
    -/
    letI := nonempty_subtype.mpr (nonempty_lifts x)
    /-
      case refine_3
      ι : Type uι
      inst✝³ : Fintype ι
      𝕜 : Type u𝕜
      inst✝² : NontriviallyNormedField 𝕜
      E : ι → Type uE
      inst✝¹ : (i : ι) → SeminormedAddCommGroup (E i)
      inst✝ : (i : ι) → NormedSpace 𝕜 (E i)
      a : 𝕜
      x : PiTensorProduct 𝕜 fun i => E i
      this : Nonempty (Subtype (Membership.mem x.lifts)) := nonempty_subtype.mpr (Pi …
      ⊢ LE.le ((fun x => iInf fun p => PiTensorProduct.projectiveSeminormAux ↑p) (HS …
    -/
    rw [Real.mul_iInf_of_nonneg (norm_nonneg _)]
    /-
      case refine_3
      ι : Type uι
      inst✝³ : Fintype ι
      𝕜 : Type u𝕜
      inst✝² : NontriviallyNormedField 𝕜
      E : ι → Type uE
      inst✝¹ : (i : ι) → SeminormedAddCommGroup (E i)
      inst✝ : (i : ι) → NormedSpace 𝕜 (E i)
      a : 𝕜
      x : PiTensorProduct 𝕜 fun i => E i
      this : Nonempty (Subtype (Membership.mem x.lifts)) := nonempty_subtype.mpr (Pi …
      ⊢ LE.le ((fun x => iInf fun p => PiTensorProduct.projectiveSeminormAux ↑p) (HS …
    -/
    refine le_ciInf ?_
    /-
      case refine_3
      ι : Type uι
      inst✝³ : Fintype ι
      𝕜 : Type u𝕜
      inst✝² : NontriviallyNormedField 𝕜
      E : ι → Type uE
      inst✝¹ : (i : ι) → SeminormedAddCommGroup (E i)
      inst✝ : (i : ι) → NormedSpace 𝕜 (E i)
      a : 𝕜
      x : PiTensorProduct 𝕜 fun i => E i
      this : Nonempty (Subtype (Membership.mem x.lifts)) := nonempty_subtype.mpr (Pi …
      ⊢ ∀ (x_1 : ↑x.lifts), LE.le ((fun x => iInf fun p => PiTensorProduct.projectiv …
    -/
    intro p
    /-
      case refine_3
      ι : Type uι
      inst✝³ : Fintype ι
      𝕜 : Type u𝕜
      inst✝² : NontriviallyNormedField 𝕜
      E : ι → Type uE
      inst✝¹ : (i : ι) → SeminormedAddCommGroup (E i)
      inst✝ : (i : ι) → NormedSpace 𝕜 (E i)
      a : 𝕜
      x : PiTensorProduct 𝕜 fun i => E i
      this : Nonempty (Subtype (Membership.mem x.lifts)) := nonempty_subtype.mpr (Pi …
      p : ↑x.lifts
      ⊢ LE.le ((fun x => iInf fun p => PiTensorProduct.projectiveSeminormAux ↑p) (HS …
    -/
    rw [← projectiveSeminormAux_smul]
    exact ciInf_le_of_le (bddBelow_projectiveSemiNormAux _)
      ⟨(List.map (fun y ↦ (a * y.1, y.2)) p.1), lifts_smul p.2 a⟩ (le_refl _)


theorem projectiveSeminorm_apply (x : ⨂[𝕜] i, E i) :
    projectiveSeminorm x = iInf (fun (p : lifts x) ↦ projectiveSeminormAux p.1) := rfl


theorem projectiveSeminorm_tprod_le (m : Π i, E i) :
    projectiveSeminorm (⨂ₜ[𝕜] i, m i) ≤ ∏ i, ‖m i‖ := by
  /-
    ι : Type uι
    inst✝³ : Fintype ι
    𝕜 : Type u𝕜
    inst✝² : NontriviallyNormedField 𝕜
    E : ι → Type uE
    inst✝¹ : (i : ι) → SeminormedAddCommGroup (E i)
    inst✝ : (i : ι) → NormedSpace 𝕜 (E i)
    m : (i : ι) → E i
    ⊢ LE.le (PiTensorProduct.projectiveSeminorm ((PiTensorProduct.tprod 𝕜) fun i = …
  -/
  rw [projectiveSeminorm_apply]
  /-
    ι : Type uι
    inst✝³ : Fintype ι
    𝕜 : Type u𝕜
    inst✝² : NontriviallyNormedField 𝕜
    E : ι → Type uE
    inst✝¹ : (i : ι) → SeminormedAddCommGroup (E i)
    inst✝ : (i : ι) → NormedSpace 𝕜 (E i)
    m : (i : ι) → E i
    ⊢ LE.le (iInf fun p => PiTensorProduct.projectiveSeminormAux ↑p) (Finset.univ. …
  -/
  convert ciInf_le (bddBelow_projectiveSemiNormAux _) ⟨[((1 : 𝕜), m)] ,?_⟩
  · simp only [projectiveSeminormAux, Function.comp_apply, List.map_cons, norm_one, one_mul,
    List.map_nil, List.sum_cons, List.sum_nil, add_zero]
    /-
      case convert_2
      ι : Type uι
      inst✝³ : Fintype ι
      𝕜 : Type u𝕜
      inst✝² : NontriviallyNormedField 𝕜
      E : ι → Type uE
      inst✝¹ : (i : ι) → SeminormedAddCommGroup (E i)
      inst✝ : (i : ι) → NormedSpace 𝕜 (E i)
      m : (i : ι) → E i
      ⊢ Membership.mem ((PiTensorProduct.tprod 𝕜) fun i => m i).lifts (List.cons { f …
    -/
  · rw [mem_lifts_iff, List.map_singleton, List.sum_singleton, one_smul]
    /-
      🎉 no goals
    -/


theorem norm_eval_le_projectiveSeminorm (x : ⨂[𝕜] i, E i) (G : Type*) [SeminormedAddCommGroup G]
    [NormedSpace 𝕜 G] (f : ContinuousMultilinearMap 𝕜 E G) :
    ‖lift f.toMultilinearMap x‖ ≤ projectiveSeminorm x * ‖f‖ := by
  /-
    ι : Type uι
    inst✝⁵ : Fintype ι
    𝕜 : Type u𝕜
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : ι → Type uE
    inst✝³ : (i : ι) → SeminormedAddCommGroup (E i)
    inst✝² : (i : ι) → NormedSpace 𝕜 (E i)
    x : PiTensorProduct 𝕜 fun i => E i
    G : Type u_1
    inst✝¹ : SeminormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    f : ContinuousMultilinearMap 𝕜 E G
    ⊢ LE.le (Norm.norm ((PiTensorProduct.lift f.toMultilinearMap) x)) (HMul.hMul ( …
  -/
  letI := nonempty_subtype.mpr (nonempty_lifts x)
  /-
    ι : Type uι
    inst✝⁵ : Fintype ι
    𝕜 : Type u𝕜
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : ι → Type uE
    inst✝³ : (i : ι) → SeminormedAddCommGroup (E i)
    inst✝² : (i : ι) → NormedSpace 𝕜 (E i)
    x : PiTensorProduct 𝕜 fun i => E i
    G : Type u_1
    inst✝¹ : SeminormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    f : ContinuousMultilinearMap 𝕜 E G
    this : Nonempty (Subtype (Membership.mem x.lifts)) := nonempty_subtype.mpr (Pi …
    ⊢ LE.le (Norm.norm ((PiTensorProduct.lift f.toMultilinearMap) x)) (HMul.hMul ( …
  -/
  rw [projectiveSeminorm_apply, Real.iInf_mul_of_nonneg (norm_nonneg _), projectiveSeminormAux]
  /-
    ι : Type uι
    inst✝⁵ : Fintype ι
    𝕜 : Type u𝕜
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : ι → Type uE
    inst✝³ : (i : ι) → SeminormedAddCommGroup (E i)
    inst✝² : (i : ι) → NormedSpace 𝕜 (E i)
    x : PiTensorProduct 𝕜 fun i => E i
    G : Type u_1
    inst✝¹ : SeminormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    f : ContinuousMultilinearMap 𝕜 E G
    this : Nonempty (Subtype (Membership.mem x.lifts)) := nonempty_subtype.mpr (Pi …
    ⊢ LE.le (Norm.norm ((PiTensorProduct.lift f.toMultilinearMap) x)) (iInf fun i  …
  -/
  refine le_ciInf ?_
  /-
    ι : Type uι
    inst✝⁵ : Fintype ι
    𝕜 : Type u𝕜
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : ι → Type uE
    inst✝³ : (i : ι) → SeminormedAddCommGroup (E i)
    inst✝² : (i : ι) → NormedSpace 𝕜 (E i)
    x : PiTensorProduct 𝕜 fun i => E i
    G : Type u_1
    inst✝¹ : SeminormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    f : ContinuousMultilinearMap 𝕜 E G
    this : Nonempty (Subtype (Membership.mem x.lifts)) := nonempty_subtype.mpr (Pi …
    ⊢ ∀ (x_1 : ↑x.lifts), LE.le (Norm.norm ((PiTensorProduct.lift f.toMultilinearM …
  -/
  intro ⟨p, hp⟩
  /-
    ι : Type uι
    inst✝⁵ : Fintype ι
    𝕜 : Type u𝕜
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : ι → Type uE
    inst✝³ : (i : ι) → SeminormedAddCommGroup (E i)
    inst✝² : (i : ι) → NormedSpace 𝕜 (E i)
    x : PiTensorProduct 𝕜 fun i => E i
    G : Type u_1
    inst✝¹ : SeminormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    f : ContinuousMultilinearMap 𝕜 E G
    this : Nonempty (Subtype (Membership.mem x.lifts)) := nonempty_subtype.mpr (Pi …
    p : FreeAddMonoid (Prod 𝕜 ((i : ι) → E i))
    hp : Membership.mem x.lifts p
    ⊢ LE.le (Norm.norm ((PiTensorProduct.lift f.toMultilinearMap) x)) (HMul.hMul ( …
  -/
  rw [mem_lifts_iff] at hp
  /-
    ι : Type uι
    inst✝⁵ : Fintype ι
    𝕜 : Type u𝕜
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : ι → Type uE
    inst✝³ : (i : ι) → SeminormedAddCommGroup (E i)
    inst✝² : (i : ι) → NormedSpace 𝕜 (E i)
    x : PiTensorProduct 𝕜 fun i => E i
    G : Type u_1
    inst✝¹ : SeminormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    f : ContinuousMultilinearMap 𝕜 E G
    this : Nonempty (Subtype (Membership.mem x.lifts)) := nonempty_subtype.mpr (Pi …
    p : FreeAddMonoid (Prod 𝕜 ((i : ι) → E i))
    hp✝ : Membership.mem x.lifts p
    hp : Eq (List.map (fun x => HSMul.hSMul x.1 ((PiTensorProduct.tprod 𝕜) fun i = …
    ⊢ LE.le (Norm.norm ((PiTensorProduct.lift f.toMultilinearMap) x)) (HMul.hMul ( …
  -/
  conv_lhs => rw [← hp, ← List.sum_map_hom, ← Multiset.sum_coe]
  /-
    ι : Type uι
    inst✝⁵ : Fintype ι
    𝕜 : Type u𝕜
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : ι → Type uE
    inst✝³ : (i : ι) → SeminormedAddCommGroup (E i)
    inst✝² : (i : ι) → NormedSpace 𝕜 (E i)
    x : PiTensorProduct 𝕜 fun i => E i
    G : Type u_1
    inst✝¹ : SeminormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    f : ContinuousMultilinearMap 𝕜 E G
    this : Nonempty (Subtype (Membership.mem x.lifts)) := nonempty_subtype.mpr (Pi …
    p : FreeAddMonoid (Prod 𝕜 ((i : ι) → E i))
    hp✝ : Membership.mem x.lifts p
    hp : Eq (List.map (fun x => HSMul.hSMul x.1 ((PiTensorProduct.tprod 𝕜) fun i = …
    ⊢ LE.le (Norm.norm (↑(List.map (Function.comp ⇑(PiTensorProduct.lift f.toMulti …
  -/
  refine le_trans (norm_multiset_sum_le _) ?_
  simp only [tprodCoeff_eq_smul_tprod, Multiset.map_coe, List.map_map, Multiset.sum_coe,
    Function.comp_apply]
  /-
    ι : Type uι
    inst✝⁵ : Fintype ι
    𝕜 : Type u𝕜
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : ι → Type uE
    inst✝³ : (i : ι) → SeminormedAddCommGroup (E i)
    inst✝² : (i : ι) → NormedSpace 𝕜 (E i)
    x : PiTensorProduct 𝕜 fun i => E i
    G : Type u_1
    inst✝¹ : SeminormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    f : ContinuousMultilinearMap 𝕜 E G
    this : Nonempty (Subtype (Membership.mem x.lifts)) := nonempty_subtype.mpr (Pi …
    p : FreeAddMonoid (Prod 𝕜 ((i : ι) → E i))
    hp✝ : Membership.mem x.lifts p
    hp : Eq (List.map (fun x => HSMul.hSMul x.1 ((PiTensorProduct.tprod 𝕜) fun i = …
    ⊢ LE.le (List.map (Function.comp (fun x => Norm.norm x) (Function.comp ⇑(PiTen …
  -/
  rw [mul_comm, ← smul_eq_mul, List.smul_sum]
  /-
    ι : Type uι
    inst✝⁵ : Fintype ι
    𝕜 : Type u𝕜
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : ι → Type uE
    inst✝³ : (i : ι) → SeminormedAddCommGroup (E i)
    inst✝² : (i : ι) → NormedSpace 𝕜 (E i)
    x : PiTensorProduct 𝕜 fun i => E i
    G : Type u_1
    inst✝¹ : SeminormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    f : ContinuousMultilinearMap 𝕜 E G
    this : Nonempty (Subtype (Membership.mem x.lifts)) := nonempty_subtype.mpr (Pi …
    p : FreeAddMonoid (Prod 𝕜 ((i : ι) → E i))
    hp✝ : Membership.mem x.lifts p
    hp : Eq (List.map (fun x => HSMul.hSMul x.1 ((PiTensorProduct.tprod 𝕜) fun i = …
    ⊢ LE.le (List.map (Function.comp (fun x => Norm.norm x) (Function.comp ⇑(PiTen …
  -/
  refine List.Forall₂.sum_le_sum ?_
  simp only [smul_eq_mul, List.map_map, List.forall₂_map_right_iff, Function.comp_apply,
    List.forall₂_map_left_iff, map_smul, lift.tprod, ContinuousMultilinearMap.coe_coe,
    List.forall₂_same, Prod.forall]
  /-
    ι : Type uι
    inst✝⁵ : Fintype ι
    𝕜 : Type u𝕜
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : ι → Type uE
    inst✝³ : (i : ι) → SeminormedAddCommGroup (E i)
    inst✝² : (i : ι) → NormedSpace 𝕜 (E i)
    x : PiTensorProduct 𝕜 fun i => E i
    G : Type u_1
    inst✝¹ : SeminormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    f : ContinuousMultilinearMap 𝕜 E G
    this : Nonempty (Subtype (Membership.mem x.lifts)) := nonempty_subtype.mpr (Pi …
    p : FreeAddMonoid (Prod 𝕜 ((i : ι) → E i))
    hp✝ : Membership.mem x.lifts p
    hp : Eq (List.map (fun x => HSMul.hSMul x.1 ((PiTensorProduct.tprod 𝕜) fun i = …
    ⊢ ∀ (a : 𝕜) (b : (i : ι) → E i), Membership.mem p { fst := a, snd := b } → LE. …
  -/
  intro a m _
  /-
    ι : Type uι
    inst✝⁵ : Fintype ι
    𝕜 : Type u𝕜
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : ι → Type uE
    inst✝³ : (i : ι) → SeminormedAddCommGroup (E i)
    inst✝² : (i : ι) → NormedSpace 𝕜 (E i)
    x : PiTensorProduct 𝕜 fun i => E i
    G : Type u_1
    inst✝¹ : SeminormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    f : ContinuousMultilinearMap 𝕜 E G
    this : Nonempty (Subtype (Membership.mem x.lifts)) := nonempty_subtype.mpr (Pi …
    p : FreeAddMonoid (Prod 𝕜 ((i : ι) → E i))
    hp✝ : Membership.mem x.lifts p
    hp : Eq (List.map (fun x => HSMul.hSMul x.1 ((PiTensorProduct.tprod 𝕜) fun i = …
    a : 𝕜
    m : (i : ι) → E i
    a✝ : Membership.mem p { fst := a, snd := m }
    ⊢ LE.le (Norm.norm (HSMul.hSMul a (f fun i => m i))) (HMul.hMul (Norm.norm f)  …
  -/
  rw [norm_smul, ← mul_assoc, mul_comm ‖f‖ _, mul_assoc]
  /-
    ι : Type uι
    inst✝⁵ : Fintype ι
    𝕜 : Type u𝕜
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : ι → Type uE
    inst✝³ : (i : ι) → SeminormedAddCommGroup (E i)
    inst✝² : (i : ι) → NormedSpace 𝕜 (E i)
    x : PiTensorProduct 𝕜 fun i => E i
    G : Type u_1
    inst✝¹ : SeminormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    f : ContinuousMultilinearMap 𝕜 E G
    this : Nonempty (Subtype (Membership.mem x.lifts)) := nonempty_subtype.mpr (Pi …
    p : FreeAddMonoid (Prod 𝕜 ((i : ι) → E i))
    hp✝ : Membership.mem x.lifts p
    hp : Eq (List.map (fun x => HSMul.hSMul x.1 ((PiTensorProduct.tprod 𝕜) fun i = …
    a : 𝕜
    m : (i : ι) → E i
    a✝ : Membership.mem p { fst := a, snd := m }
    ⊢ LE.le (HMul.hMul (Norm.norm a) (Norm.norm (f fun i => m i))) (HMul.hMul (Nor …
  -/
  exact mul_le_mul_of_nonneg_left (f.le_opNorm _) (norm_nonneg _)
  /-
    🎉 no goals
  -/


