theorem proj_diagonal (i : n) (w : n → R) : (proj i).comp (toLin' (diagonal w)) = w i • proj i :=
  LinearMap.ext fun _ => mulVec_diagonal _ _ _


theorem diagonal_comp_single (w : n → R) (i : n) :
    (diagonal w).toLin'.comp (LinearMap.single R (fun _ : n => R) i) =
      w i • LinearMap.single R (fun _ : n => R) i :=
  LinearMap.ext fun x => (diagonal_mulVec_single w _ _).trans (Pi.single_smul' i (w i) x)


set_option linter.deprecated false in
@[deprecated diagonal_comp_single (since := "2024-08-09")]
theorem diagonal_comp_stdBasis (w : n → R) (i : n) :
    (diagonal w).toLin'.comp (LinearMap.stdBasis R (fun _ : n => R) i) =
      w i • LinearMap.stdBasis R (fun _ : n => R) i :=
  LinearMap.ext fun x => (diagonal_mulVec_single w _ _).trans (Pi.single_smul' i (w i) x)


theorem diagonal_toLin' (w : n → R) :
    toLin' (diagonal w) = LinearMap.pi fun i => w i • LinearMap.proj i :=
  LinearMap.ext fun _ => funext fun _ => mulVec_diagonal _ _ _


theorem ker_diagonal_toLin' [DecidableEq m] (w : m → K) :
    ker (toLin' (diagonal w)) =
      ⨆ i ∈ { i | w i = 0 }, LinearMap.range (LinearMap.single K (fun _ => K) i) := by
  /-
    m : Type u_1
    inst✝² : Fintype m
    K : Type u
    inst✝¹ : Semifield K
    inst✝ : DecidableEq m
    w : m → K
    ⊢ Eq (LinearMap.ker (Matrix.toLin' (Matrix.diagonal w))) (iSup fun i => iSup f …
  -/
  rw [← comap_bot, ← iInf_ker_proj, comap_iInf]
  /-
    m : Type u_1
    inst✝² : Fintype m
    K : Type u
    inst✝¹ : Semifield K
    inst✝ : DecidableEq m
    w : m → K
    ⊢ Eq (iInf fun i => Submodule.comap (Matrix.toLin' (Matrix.diagonal w)) (Linea …
  -/
  have := fun i : m => ker_comp (toLin' (diagonal w)) (proj i)
  /-
    m : Type u_1
    inst✝² : Fintype m
    K : Type u
    inst✝¹ : Semifield K
    inst✝ : DecidableEq m
    w : m → K
    this : ∀ (i : m), Eq (LinearMap.ker ((LinearMap.proj i).comp (Matrix.toLin' (M …
    ⊢ Eq (iInf fun i => Submodule.comap (Matrix.toLin' (Matrix.diagonal w)) (Linea …
  -/
  simp only [comap_iInf, ← this, proj_diagonal, ker_smul']
  /-
    m : Type u_1
    inst✝² : Fintype m
    K : Type u
    inst✝¹ : Semifield K
    inst✝ : DecidableEq m
    w : m → K
    this : ∀ (i : m), Eq (LinearMap.ker ((LinearMap.proj i).comp (Matrix.toLin' (M …
    ⊢ Eq (iInf fun i => iInf fun x => LinearMap.ker (LinearMap.proj i)) (iSup fun  …
  -/
  have : univ ⊆ { i : m | w i = 0 } ∪ { i : m | w i = 0 }ᶜ := by rw [Set.union_compl_self]
  exact (iSup_range_single_eq_iInf_ker_proj K (fun _ : m => K) disjoint_compl_right this
    (Set.toFinite _)).symm


theorem range_diagonal [DecidableEq m] (w : m → K) :
    LinearMap.range (toLin' (diagonal w)) =
      ⨆ i ∈ { i | w i ≠ 0 }, LinearMap.range (LinearMap.single K (fun _ => K) i) := by
  /-
    m : Type u_1
    inst✝² : Fintype m
    K : Type u
    inst✝¹ : Semifield K
    inst✝ : DecidableEq m
    w : m → K
    ⊢ Eq (LinearMap.range (Matrix.toLin' (Matrix.diagonal w))) (iSup fun i => iSup …
  -/
  dsimp only [mem_setOf_eq]
  /-
    m : Type u_1
    inst✝² : Fintype m
    K : Type u
    inst✝¹ : Semifield K
    inst✝ : DecidableEq m
    w : m → K
    ⊢ Eq (LinearMap.range (Matrix.toLin' (Matrix.diagonal w))) (iSup fun i => iSup …
  -/
  rw [← Submodule.map_top, ← iSup_range_single, Submodule.map_iSup]
  /-
    m : Type u_1
    inst✝² : Fintype m
    K : Type u
    inst✝¹ : Semifield K
    inst✝ : DecidableEq m
    w : m → K
    ⊢ Eq (iSup fun i => Submodule.map (Matrix.toLin' (Matrix.diagonal w)) (LinearM …
  -/
  congr; funext i
  /-
    case e_s.h
    m : Type u_1
    inst✝² : Fintype m
    K : Type u
    inst✝¹ : Semifield K
    inst✝ : DecidableEq m
    w : m → K
    i : m
    ⊢ Eq (Submodule.map (Matrix.toLin' (Matrix.diagonal w)) (LinearMap.range (Line …
  -/
  rw [← LinearMap.range_comp, diagonal_comp_single, ← range_smul']
  /-
    🎉 no goals
  -/


theorem rank_diagonal [DecidableEq m] [DecidableEq K] (w : m → K) :
    LinearMap.rank (toLin' (diagonal w)) = Fintype.card { i // w i ≠ 0 } := by
  /-
    m : Type u_1
    inst✝³ : Fintype m
    K : Type u
    inst✝² : Field K
    inst✝¹ : DecidableEq m
    inst✝ : DecidableEq K
    w : m → K
    ⊢ Eq (Matrix.toLin' (Matrix.diagonal w)).rank ↑(Fintype.card (Subtype fun i => …
  -/
  have hu : univ ⊆ { i : m | w i = 0 }ᶜ ∪ { i : m | w i = 0 } := by rw [Set.compl_union_self]
  /-
    m : Type u_1
    inst✝³ : Fintype m
    K : Type u
    inst✝² : Field K
    inst✝¹ : DecidableEq m
    inst✝ : DecidableEq K
    w : m → K
    hu : HasSubset.Subset Set.univ (Union.union (HasCompl.compl (setOf fun i => Eq …
    ⊢ Eq (Matrix.toLin' (Matrix.diagonal w)).rank ↑(Fintype.card (Subtype fun i => …
  -/
  have hd : Disjoint { i : m | w i ≠ 0 } { i : m | w i = 0 } := disjoint_compl_left
  /-
    m : Type u_1
    inst✝³ : Fintype m
    K : Type u
    inst✝² : Field K
    inst✝¹ : DecidableEq m
    inst✝ : DecidableEq K
    w : m → K
    hu : HasSubset.Subset Set.univ (Union.union (HasCompl.compl (setOf fun i => Eq …
    hd : Disjoint (setOf fun i => Ne (w i) 0) (setOf fun i => Eq (w i) 0)
    ⊢ Eq (Matrix.toLin' (Matrix.diagonal w)).rank ↑(Fintype.card (Subtype fun i => …
  -/
  have B₁ := iSup_range_single_eq_iInf_ker_proj K (fun _ : m => K) hd hu (Set.toFinite _)
  /-
    m : Type u_1
    inst✝³ : Fintype m
    K : Type u
    inst✝² : Field K
    inst✝¹ : DecidableEq m
    inst✝ : DecidableEq K
    w : m → K
    hu : HasSubset.Subset Set.univ (Union.union (HasCompl.compl (setOf fun i => Eq …
    hd : Disjoint (setOf fun i => Ne (w i) 0) (setOf fun i => Eq (w i) 0)
    B₁ : Eq (iSup fun i => iSup fun h => LinearMap.range (LinearMap.single K (fun  …
    ⊢ Eq (Matrix.toLin' (Matrix.diagonal w)).rank ↑(Fintype.card (Subtype fun i => …
  -/
  have B₂ := iInfKerProjEquiv K (fun _ ↦ K) hd hu
  /-
    m : Type u_1
    inst✝³ : Fintype m
    K : Type u
    inst✝² : Field K
    inst✝¹ : DecidableEq m
    inst✝ : DecidableEq K
    w : m → K
    hu : HasSubset.Subset Set.univ (Union.union (HasCompl.compl (setOf fun i => Eq …
    hd : Disjoint (setOf fun i => Ne (w i) 0) (setOf fun i => Eq (w i) 0)
    B₁ : Eq (iSup fun i => iSup fun h => LinearMap.range (LinearMap.single K (fun  …
    B₂ : LinearEquiv (RingHom.id K) (Subtype fun x => Membership.mem (iInf fun i = …
    ⊢ Eq (Matrix.toLin' (Matrix.diagonal w)).rank ↑(Fintype.card (Subtype fun i => …
  -/
  rw [LinearMap.rank, range_diagonal, B₁, ← @rank_fun' K]
  /-
    m : Type u_1
    inst✝³ : Fintype m
    K : Type u
    inst✝² : Field K
    inst✝¹ : DecidableEq m
    inst✝ : DecidableEq K
    w : m → K
    hu : HasSubset.Subset Set.univ (Union.union (HasCompl.compl (setOf fun i => Eq …
    hd : Disjoint (setOf fun i => Ne (w i) 0) (setOf fun i => Eq (w i) 0)
    B₁ : Eq (iSup fun i => iSup fun h => LinearMap.range (LinearMap.single K (fun  …
    B₂ : LinearEquiv (RingHom.id K) (Subtype fun x => Membership.mem (iInf fun i = …
    ⊢ Eq (Module.rank K (Subtype fun x => Membership.mem (iInf fun i => iInf fun h …
  -/
  apply LinearEquiv.rank_eq
  /-
    case f
    m : Type u_1
    inst✝³ : Fintype m
    K : Type u
    inst✝² : Field K
    inst✝¹ : DecidableEq m
    inst✝ : DecidableEq K
    w : m → K
    hu : HasSubset.Subset Set.univ (Union.union (HasCompl.compl (setOf fun i => Eq …
    hd : Disjoint (setOf fun i => Ne (w i) 0) (setOf fun i => Eq (w i) 0)
    B₁ : Eq (iSup fun i => iSup fun h => LinearMap.range (LinearMap.single K (fun  …
    B₂ : LinearEquiv (RingHom.id K) (Subtype fun x => Membership.mem (iInf fun i = …
    ⊢ LinearEquiv (RingHom.id K) (Subtype fun x => Membership.mem (iInf fun i => i …
  -/
  apply B₂
  /-
    🎉 no goals
  -/


