lemma dist_left_le_of_mem_uIcc {x y z : ℝ} (h : y ∈ uIcc x z) : dist x y ≤ dist x z := by
  /-
    x y z : Real
    h : Membership.mem (Set.uIcc x z) y
    ⊢ LE.le (Dist.dist x y) (Dist.dist x z)
  -/
  simpa only [dist_comm x] using abs_sub_left_of_mem_uIcc h
  /-
    🎉 no goals
  -/


lemma dist_right_le_of_mem_uIcc {x y z : ℝ} (h : y ∈ uIcc x z) : dist y z ≤ dist x z := by
  /-
    x y z : Real
    h : Membership.mem (Set.uIcc x z) y
    ⊢ LE.le (Dist.dist y z) (Dist.dist x z)
  -/
  simpa only [dist_comm _ z] using abs_sub_right_of_mem_uIcc h
  /-
    🎉 no goals
  -/


lemma dist_le_of_mem_uIcc {x y x' y' : ℝ} (hx : x ∈ uIcc x' y') (hy : y ∈ uIcc x' y') :
    dist x y ≤ dist x' y' :=
                                                         /-
                                                           x y x' y' : Real
                                                           hx : Membership.mem (Set.uIcc x' y') x
                                                           hy : Membership.mem (Set.uIcc x' y') y
                                                           ⊢ Membership.mem (Set.uIcc y' x') y
                                                         -/
                                                         /-
                                                           🎉 no goals
                                                         -/
  abs_sub_le_of_uIcc_subset_uIcc <| uIcc_subset_uIcc (by rwa [uIcc_comm]) (by rwa [uIcc_comm])
                                                                              /-
                                                                                🎉 no goals
                                                                              -/


lemma dist_le_of_mem_Icc {x y x' y' : ℝ} (hx : x ∈ Icc x' y') (hy : y ∈ Icc x' y') :
    dist x y ≤ y' - x' := by
  simpa only [Real.dist_eq, abs_of_nonpos (sub_nonpos.2 <| hx.1.trans hx.2), neg_sub] using
    Real.dist_le_of_mem_uIcc (Icc_subset_uIcc hx) (Icc_subset_uIcc hy)


lemma dist_le_of_mem_Icc_01 {x y : ℝ} (hx : x ∈ Icc (0 : ℝ) 1) (hy : y ∈ Icc (0 : ℝ) 1) :
                       /-
                         x y : Real
                         hx : Membership.mem (Set.Icc 0 1) x
                         hy : Membership.mem (Set.Icc 0 1) y
                         ⊢ LE.le (Dist.dist x y) 1
                       -/
    dist x y ≤ 1 := by simpa only [sub_zero] using Real.dist_le_of_mem_Icc hx hy
                       /-
                         🎉 no goals
                       -/


lemma dist_le_of_mem_pi_Icc (hx : x ∈ Icc x' y') (hy : y ∈ Icc x' y') : dist x y ≤ dist x' y' := by
  refine (dist_pi_le_iff dist_nonneg).2 fun b =>
                                                                         /-
                                                                           case refine_1
                                                                           ι : Type u_1
                                                                           inst✝ : Fintype ι
                                                                           x y x' y' : ι → Real
                                                                           hx : Membership.mem (Set.Icc x' y') x
                                                                           hy : Membership.mem (Set.Icc x' y') y
                                                                           b : ι
                                                                           ⊢ Membership.mem (Set.uIcc (x' b) (y' b)) (x b)
                                                                         -/
    (Real.dist_le_of_mem_uIcc ?_ ?_).trans (dist_le_pi_dist x' y' b) <;> refine Icc_subset_uIcc ?_
  /-
    case refine_1
    ι : Type u_1
    inst✝ : Fintype ι
    x y x' y' : ι → Real
    hx : Membership.mem (Set.Icc x' y') x
    hy : Membership.mem (Set.Icc x' y') y
    b : ι
    ⊢ Membership.mem (Set.Icc (x' b) (y' b)) (x b)
  -/
  exacts [⟨hx.1 _, hx.2 _⟩, ⟨hy.1 _, hy.2 _⟩]
  /-
    🎉 no goals
  -/


