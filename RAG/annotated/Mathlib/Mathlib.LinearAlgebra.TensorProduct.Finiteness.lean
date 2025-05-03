/-- For any element `x` of `M ⊗[R] N`, there exists a (finite) multiset `{ (m_i, n_i) }`
of `M × N`, such that `x` is equal to the sum of `m_i ⊗ₜ[R] n_i`. -/
theorem exists_multiset (x : M ⊗[R] N) :
    ∃ S : Multiset (M × N), x = (S.map fun i ↦ i.1 ⊗ₜ[R] i.2).sum := by
  induction x with
  | zero => exact ⟨0, by simp⟩
  | tmul x y => exact ⟨{(x, y)}, by simp⟩
  | add x y hx hy =>
    obtain ⟨Sx, hx⟩ := hx
    obtain ⟨Sy, hy⟩ := hy
    exact ⟨Sx + Sy, by rw [Multiset.map_add, Multiset.sum_add, hx, hy]⟩


/-- For any element `x` of `M ⊗[R] N`, there exists a finite subset `{ (m_i, n_i) }`
of `M × N` such that each `m_i` is distinct (we represent it as an element of `M →₀ N`),
such that `x` is equal to the sum of `m_i ⊗ₜ[R] n_i`. -/
theorem exists_finsupp_left (x : M ⊗[R] N) :
    ∃ S : M →₀ N, x = S.sum fun m n ↦ m ⊗ₜ[R] n := by
  induction x with
  | zero => exact ⟨0, by simp⟩
  | tmul x y => exact ⟨Finsupp.single x y, by simp⟩
  | add x y hx hy =>
    obtain ⟨Sx, hx⟩ := hx
    obtain ⟨Sy, hy⟩ := hy
    use Sx + Sy
    rw [hx, hy]
    exact (Finsupp.sum_add_index' (by simp) TensorProduct.tmul_add).symm


/-- For any element `x` of `M ⊗[R] N`, there exists a finite subset `{ (m_i, n_i) }`
of `M × N` such that each `n_i` is distinct (we represent it as an element of `N →₀ M`),
such that `x` is equal to the sum of `m_i ⊗ₜ[R] n_i`. -/
theorem exists_finsupp_right (x : M ⊗[R] N) :
    ∃ S : N →₀ M, x = S.sum fun n m ↦ m ⊗ₜ[R] n := by
  /-
    R : Type u_1
    M : Type u_2
    N : Type u_3
    inst✝⁴ : CommSemiring R
    inst✝³ : AddCommMonoid M
    inst✝² : AddCommMonoid N
    inst✝¹ : Module R M
    inst✝ : Module R N
    x : TensorProduct R M N
    ⊢ Exists fun S => Eq x (S.sum fun n m => TensorProduct.tmul R m n)
  -/
  obtain ⟨S, h⟩ := exists_finsupp_left (TensorProduct.comm R M N x)
  /-
    case intro
    R : Type u_1
    M : Type u_2
    N : Type u_3
    inst✝⁴ : CommSemiring R
    inst✝³ : AddCommMonoid M
    inst✝² : AddCommMonoid N
    inst✝¹ : Module R M
    inst✝ : Module R N
    x : TensorProduct R M N
    S : Finsupp N M
    h : Eq ((TensorProduct.comm R M N) x) (S.sum fun m n => TensorProduct.tmul R m …
    ⊢ Exists fun S => Eq x (S.sum fun n m => TensorProduct.tmul R m n)
  -/
  refine ⟨S, (TensorProduct.comm R M N).injective ?_⟩
  /-
    case intro
    R : Type u_1
    M : Type u_2
    N : Type u_3
    inst✝⁴ : CommSemiring R
    inst✝³ : AddCommMonoid M
    inst✝² : AddCommMonoid N
    inst✝¹ : Module R M
    inst✝ : Module R N
    x : TensorProduct R M N
    S : Finsupp N M
    h : Eq ((TensorProduct.comm R M N) x) (S.sum fun m n => TensorProduct.tmul R m …
    ⊢ Eq ((TensorProduct.comm R M N) x) ((TensorProduct.comm R M N) (S.sum fun n m …
  -/
  simp_rw [h, Finsupp.sum, map_sum, comm_tmul]
  /-
    🎉 no goals
  -/


/-- For any element `x` of `M ⊗[R] N`, there exists a finite subset `{ (m_i, n_i) }`
of `M × N`, such that `x` is equal to the sum of `m_i ⊗ₜ[R] n_i`. -/
theorem exists_finset (x : M ⊗[R] N) :
    ∃ S : Finset (M × N), x = S.sum fun i ↦ i.1 ⊗ₜ[R] i.2 := by
  /-
    R : Type u_1
    M : Type u_2
    N : Type u_3
    inst✝⁴ : CommSemiring R
    inst✝³ : AddCommMonoid M
    inst✝² : AddCommMonoid N
    inst✝¹ : Module R M
    inst✝ : Module R N
    x : TensorProduct R M N
    ⊢ Exists fun S => Eq x (S.sum fun i => TensorProduct.tmul R i.1 i.2)
  -/
  obtain ⟨S, h⟩ := exists_finsupp_left x
  /-
    case intro
    R : Type u_1
    M : Type u_2
    N : Type u_3
    inst✝⁴ : CommSemiring R
    inst✝³ : AddCommMonoid M
    inst✝² : AddCommMonoid N
    inst✝¹ : Module R M
    inst✝ : Module R N
    x : TensorProduct R M N
    S : Finsupp M N
    h : Eq x (S.sum fun m n => TensorProduct.tmul R m n)
    ⊢ Exists fun S => Eq x (S.sum fun i => TensorProduct.tmul R i.1 i.2)
  -/
  use S.graph
  /-
    case h
    R : Type u_1
    M : Type u_2
    N : Type u_3
    inst✝⁴ : CommSemiring R
    inst✝³ : AddCommMonoid M
    inst✝² : AddCommMonoid N
    inst✝¹ : Module R M
    inst✝ : Module R N
    x : TensorProduct R M N
    S : Finsupp M N
    h : Eq x (S.sum fun m n => TensorProduct.tmul R m n)
    ⊢ Eq x (S.graph.sum fun i => TensorProduct.tmul R i.1 i.2)
  -/
  rw [h, Finsupp.sum]
  /-
    case h
    R : Type u_1
    M : Type u_2
    N : Type u_3
    inst✝⁴ : CommSemiring R
    inst✝³ : AddCommMonoid M
    inst✝² : AddCommMonoid N
    inst✝¹ : Module R M
    inst✝ : Module R N
    x : TensorProduct R M N
    S : Finsupp M N
    h : Eq x (S.sum fun m n => TensorProduct.tmul R m n)
    ⊢ Eq (S.support.sum fun a => TensorProduct.tmul R a (S a)) (S.graph.sum fun i  …
  -/
                                                         /-
                                                           🎉 no goals
                                                         -/
                                                         /-
                                                           🎉 no goals
                                                         -/
                                                         /-
                                                           🎉 no goals
                                                         -/
                                                         /-
                                                           🎉 no goals
                                                         -/
  apply Finset.sum_nbij' (fun m ↦ ⟨m, S m⟩) Prod.fst <;> simp
                                                         /-
                                                           🎉 no goals
                                                         -/


/-- For a finite subset `s` of `M ⊗[R] N`, there are finitely generated
submodules `M'` and `N'` of `M` and `N`, respectively, such that `s` is contained in the image
of `M' ⊗[R] N'` in `M ⊗[R] N`. -/
theorem exists_finite_submodule_of_finite (s : Set (M ⊗[R] N)) (hs : s.Finite) :
    ∃ (M' : Submodule R M) (N' : Submodule R N), Module.Finite R M' ∧ Module.Finite R N' ∧
      s ⊆ LinearMap.range (mapIncl M' N') := by
  /-
    R : Type u_1
    M : Type u_2
    N : Type u_3
    inst✝⁴ : CommSemiring R
    inst✝³ : AddCommMonoid M
    inst✝² : AddCommMonoid N
    inst✝¹ : Module R M
    inst✝ : Module R N
    s : Set (TensorProduct R M N)
    hs : s.Finite
    ⊢ Exists fun M' => Exists fun N' => And (Module.Finite R (Subtype fun x => Mem …
  -/
  simp_rw [Module.Finite.iff_fg]
  /-
    R : Type u_1
    M : Type u_2
    N : Type u_3
    inst✝⁴ : CommSemiring R
    inst✝³ : AddCommMonoid M
    inst✝² : AddCommMonoid N
    inst✝¹ : Module R M
    inst✝ : Module R N
    s : Set (TensorProduct R M N)
    hs : s.Finite
    ⊢ Exists fun M' => Exists fun N' => And M'.FG (And N'.FG (HasSubset.Subset s ↑ …
  -/
  refine hs.induction_on ⟨_, _, fg_bot, fg_bot, Set.empty_subset _⟩ ?_
  /-
    R : Type u_1
    M : Type u_2
    N : Type u_3
    inst✝⁴ : CommSemiring R
    inst✝³ : AddCommMonoid M
    inst✝² : AddCommMonoid N
    inst✝¹ : Module R M
    inst✝ : Module R N
    s : Set (TensorProduct R M N)
    hs : s.Finite
    ⊢ ∀ {a : TensorProduct R M N} {s : Set (TensorProduct R M N)}, Not (Membership …
  -/
  rintro a s - - ⟨M', N', hM', hN', h⟩
  /-
    case intro.intro.intro.intro
    R : Type u_1
    M : Type u_2
    N : Type u_3
    inst✝⁴ : CommSemiring R
    inst✝³ : AddCommMonoid M
    inst✝² : AddCommMonoid N
    inst✝¹ : Module R M
    inst✝ : Module R N
    s✝ : Set (TensorProduct R M N)
    hs : s✝.Finite
    a : TensorProduct R M N
    s : Set (TensorProduct R M N)
    M' : Submodule R M
    N' : Submodule R N
    hM' : M'.FG
    hN' : N'.FG
    h : HasSubset.Subset s ↑(LinearMap.range (TensorProduct.mapIncl M' N'))
    ⊢ Exists fun M' => Exists fun N' => And M'.FG (And N'.FG (HasSubset.Subset (In …
  -/
  refine TensorProduct.induction_on a ?_ (fun x y ↦ ?_) fun x y hx hy ↦ ?_
    /-
      case intro.intro.intro.intro.refine_1
      R : Type u_1
      M : Type u_2
      N : Type u_3
      inst✝⁴ : CommSemiring R
      inst✝³ : AddCommMonoid M
      inst✝² : AddCommMonoid N
      inst✝¹ : Module R M
      inst✝ : Module R N
      s✝ : Set (TensorProduct R M N)
      hs : s✝.Finite
      a : TensorProduct R M N
      s : Set (TensorProduct R M N)
      M' : Submodule R M
      N' : Submodule R N
      hM' : M'.FG
      hN' : N'.FG
      h : HasSubset.Subset s ↑(LinearMap.range (TensorProduct.mapIncl M' N'))
      ⊢ Exists fun M' => Exists fun N' => And M'.FG (And N'.FG (HasSubset.Subset (In …
    -/
  · exact ⟨M', N', hM', hN', Set.insert_subset (zero_mem _) h⟩
    /-
      🎉 no goals
    -/
  · refine ⟨_, _, hM'.sup (fg_span_singleton x),
      hN'.sup (fg_span_singleton y), Set.insert_subset ?_ fun z hz ↦ ?_⟩
    · exact ⟨⟨x, mem_sup_right (mem_span_singleton_self x)⟩ ⊗ₜ
        ⟨y, mem_sup_right (mem_span_singleton_self y)⟩, rfl⟩
      /-
        case intro.intro.intro.intro.refine_2.refine_2
        R : Type u_1
        M : Type u_2
        N : Type u_3
        inst✝⁴ : CommSemiring R
        inst✝³ : AddCommMonoid M
        inst✝² : AddCommMonoid N
        inst✝¹ : Module R M
        inst✝ : Module R N
        s✝ : Set (TensorProduct R M N)
        hs : s✝.Finite
        a : TensorProduct R M N
        s : Set (TensorProduct R M N)
        M' : Submodule R M
        N' : Submodule R N
        hM' : M'.FG
        hN' : N'.FG
        h : HasSubset.Subset s ↑(LinearMap.range (TensorProduct.mapIncl M' N'))
        x : M
        y : N
        z : TensorProduct R M N
        hz : Membership.mem s z
        ⊢ Membership.mem (↑(LinearMap.range (TensorProduct.mapIncl (Max.max M' (Submod …
      -/
    · exact range_mapIncl_mono le_sup_left le_sup_left (h hz)
      /-
        🎉 no goals
      -/
    /-
      case intro.intro.intro.intro.refine_3
      R : Type u_1
      M : Type u_2
      N : Type u_3
      inst✝⁴ : CommSemiring R
      inst✝³ : AddCommMonoid M
      inst✝² : AddCommMonoid N
      inst✝¹ : Module R M
      inst✝ : Module R N
      s✝ : Set (TensorProduct R M N)
      hs : s✝.Finite
      a : TensorProduct R M N
      s : Set (TensorProduct R M N)
      M' : Submodule R M
      N' : Submodule R N
      hM' : M'.FG
      hN' : N'.FG
      h : HasSubset.Subset s ↑(LinearMap.range (TensorProduct.mapIncl M' N'))
      x y : TensorProduct R M N
      hx : Exists fun M' => Exists fun N' => And M'.FG (And N'.FG (HasSubset.Subset  …
      hy : Exists fun M' => Exists fun N' => And M'.FG (And N'.FG (HasSubset.Subset  …
      ⊢ Exists fun M' => Exists fun N' => And M'.FG (And N'.FG (HasSubset.Subset (In …
    -/
  · obtain ⟨M₁', N₁', hM₁', hN₁', h₁⟩ := hx
    /-
      case intro.intro.intro.intro.refine_3.intro.intro.intro.intro
      R : Type u_1
      M : Type u_2
      N : Type u_3
      inst✝⁴ : CommSemiring R
      inst✝³ : AddCommMonoid M
      inst✝² : AddCommMonoid N
      inst✝¹ : Module R M
      inst✝ : Module R N
      s✝ : Set (TensorProduct R M N)
      hs : s✝.Finite
      a : TensorProduct R M N
      s : Set (TensorProduct R M N)
      M' : Submodule R M
      N' : Submodule R N
      hM' : M'.FG
      hN' : N'.FG
      h : HasSubset.Subset s ↑(LinearMap.range (TensorProduct.mapIncl M' N'))
      x y : TensorProduct R M N
      hy : Exists fun M' => Exists fun N' => And M'.FG (And N'.FG (HasSubset.Subset  …
      M₁' : Submodule R M
      N₁' : Submodule R N
      hM₁' : M₁'.FG
      hN₁' : N₁'.FG
      h₁ : HasSubset.Subset (Insert.insert x s) ↑(LinearMap.range (TensorProduct.map …
      ⊢ Exists fun M' => Exists fun N' => And M'.FG (And N'.FG (HasSubset.Subset (In …
    -/
    obtain ⟨M₂', N₂', hM₂', hN₂', h₂⟩ := hy
    /-
      case intro.intro.intro.intro.refine_3.intro.intro.intro.intro.intro.intro.intr …
      R : Type u_1
      M : Type u_2
      N : Type u_3
      inst✝⁴ : CommSemiring R
      inst✝³ : AddCommMonoid M
      inst✝² : AddCommMonoid N
      inst✝¹ : Module R M
      inst✝ : Module R N
      s✝ : Set (TensorProduct R M N)
      hs : s✝.Finite
      a : TensorProduct R M N
      s : Set (TensorProduct R M N)
      M' : Submodule R M
      N' : Submodule R N
      hM' : M'.FG
      hN' : N'.FG
      h : HasSubset.Subset s ↑(LinearMap.range (TensorProduct.mapIncl M' N'))
      x y : TensorProduct R M N
      M₁' : Submodule R M
      N₁' : Submodule R N
      hM₁' : M₁'.FG
      hN₁' : N₁'.FG
      h₁ : HasSubset.Subset (Insert.insert x s) ↑(LinearMap.range (TensorProduct.map …
      M₂' : Submodule R M
      N₂' : Submodule R N
      hM₂' : M₂'.FG
      hN₂' : N₂'.FG
      h₂ : HasSubset.Subset (Insert.insert y s) ↑(LinearMap.range (TensorProduct.map …
      ⊢ Exists fun M' => Exists fun N' => And M'.FG (And N'.FG (HasSubset.Subset (In …
    -/
    refine ⟨_, _, hM₁'.sup hM₂', hN₁'.sup hN₂', Set.insert_subset (add_mem ?_ ?_) fun z hz ↦ ?_⟩
      /-
        case intro.intro.intro.intro.refine_3.intro.intro.intro.intro.intro.intro.intr …
        R : Type u_1
        M : Type u_2
        N : Type u_3
        inst✝⁴ : CommSemiring R
        inst✝³ : AddCommMonoid M
        inst✝² : AddCommMonoid N
        inst✝¹ : Module R M
        inst✝ : Module R N
        s✝ : Set (TensorProduct R M N)
        hs : s✝.Finite
        a : TensorProduct R M N
        s : Set (TensorProduct R M N)
        M' : Submodule R M
        N' : Submodule R N
        hM' : M'.FG
        hN' : N'.FG
        h : HasSubset.Subset s ↑(LinearMap.range (TensorProduct.mapIncl M' N'))
        x y : TensorProduct R M N
        M₁' : Submodule R M
        N₁' : Submodule R N
        hM₁' : M₁'.FG
        hN₁' : N₁'.FG
        h₁ : HasSubset.Subset (Insert.insert x s) ↑(LinearMap.range (TensorProduct.map …
        M₂' : Submodule R M
        N₂' : Submodule R N
        hM₂' : M₂'.FG
        hN₂' : N₂'.FG
        h₂ : HasSubset.Subset (Insert.insert y s) ↑(LinearMap.range (TensorProduct.map …
        ⊢ Membership.mem (LinearMap.range (TensorProduct.mapIncl (Max.max M₁' M₂') (Ma …
      -/
    · exact range_mapIncl_mono le_sup_left le_sup_left (h₁ (Set.mem_insert x s))
      /-
        🎉 no goals
      -/
      /-
        case intro.intro.intro.intro.refine_3.intro.intro.intro.intro.intro.intro.intr …
        R : Type u_1
        M : Type u_2
        N : Type u_3
        inst✝⁴ : CommSemiring R
        inst✝³ : AddCommMonoid M
        inst✝² : AddCommMonoid N
        inst✝¹ : Module R M
        inst✝ : Module R N
        s✝ : Set (TensorProduct R M N)
        hs : s✝.Finite
        a : TensorProduct R M N
        s : Set (TensorProduct R M N)
        M' : Submodule R M
        N' : Submodule R N
        hM' : M'.FG
        hN' : N'.FG
        h : HasSubset.Subset s ↑(LinearMap.range (TensorProduct.mapIncl M' N'))
        x y : TensorProduct R M N
        M₁' : Submodule R M
        N₁' : Submodule R N
        hM₁' : M₁'.FG
        hN₁' : N₁'.FG
        h₁ : HasSubset.Subset (Insert.insert x s) ↑(LinearMap.range (TensorProduct.map …
        M₂' : Submodule R M
        N₂' : Submodule R N
        hM₂' : M₂'.FG
        hN₂' : N₂'.FG
        h₂ : HasSubset.Subset (Insert.insert y s) ↑(LinearMap.range (TensorProduct.map …
        ⊢ Membership.mem (LinearMap.range (TensorProduct.mapIncl (Max.max M₁' M₂') (Ma …
      -/
    · exact range_mapIncl_mono le_sup_right le_sup_right (h₂ (Set.mem_insert y s))
      /-
        🎉 no goals
      -/
      /-
        case intro.intro.intro.intro.refine_3.intro.intro.intro.intro.intro.intro.intr …
        R : Type u_1
        M : Type u_2
        N : Type u_3
        inst✝⁴ : CommSemiring R
        inst✝³ : AddCommMonoid M
        inst✝² : AddCommMonoid N
        inst✝¹ : Module R M
        inst✝ : Module R N
        s✝ : Set (TensorProduct R M N)
        hs : s✝.Finite
        a : TensorProduct R M N
        s : Set (TensorProduct R M N)
        M' : Submodule R M
        N' : Submodule R N
        hM' : M'.FG
        hN' : N'.FG
        h : HasSubset.Subset s ↑(LinearMap.range (TensorProduct.mapIncl M' N'))
        x y : TensorProduct R M N
        M₁' : Submodule R M
        N₁' : Submodule R N
        hM₁' : M₁'.FG
        hN₁' : N₁'.FG
        h₁ : HasSubset.Subset (Insert.insert x s) ↑(LinearMap.range (TensorProduct.map …
        M₂' : Submodule R M
        N₂' : Submodule R N
        hM₂' : M₂'.FG
        hN₂' : N₂'.FG
        h₂ : HasSubset.Subset (Insert.insert y s) ↑(LinearMap.range (TensorProduct.map …
        z : TensorProduct R M N
        hz : Membership.mem s z
        ⊢ Membership.mem (↑(LinearMap.range (TensorProduct.mapIncl (Max.max M₁' M₂') ( …
      -/
    · exact range_mapIncl_mono le_sup_left le_sup_left (h₁ (Set.subset_insert x s hz))
      /-
        🎉 no goals
      -/


/-- For a finite subset `s` of `M ⊗[R] N`, there exists a finitely generated
submodule `M'` of `M`, such that `s` is contained in the image
of `M' ⊗[R] N` in `M ⊗[R] N`. -/
theorem exists_finite_submodule_left_of_finite (s : Set (M ⊗[R] N)) (hs : s.Finite) :
    ∃ M' : Submodule R M, Module.Finite R M' ∧ s ⊆ LinearMap.range (M'.subtype.rTensor N) := by
  /-
    R : Type u_1
    M : Type u_2
    N : Type u_3
    inst✝⁴ : CommSemiring R
    inst✝³ : AddCommMonoid M
    inst✝² : AddCommMonoid N
    inst✝¹ : Module R M
    inst✝ : Module R N
    s : Set (TensorProduct R M N)
    hs : s.Finite
    ⊢ Exists fun M' => And (Module.Finite R (Subtype fun x => Membership.mem M' x) …
  -/
  obtain ⟨M', _, hfin, _, h⟩ := exists_finite_submodule_of_finite s hs
  /-
    case intro.intro.intro.intro
    R : Type u_1
    M : Type u_2
    N : Type u_3
    inst✝⁴ : CommSemiring R
    inst✝³ : AddCommMonoid M
    inst✝² : AddCommMonoid N
    inst✝¹ : Module R M
    inst✝ : Module R N
    s : Set (TensorProduct R M N)
    hs : s.Finite
    M' : Submodule R M
    w✝ : Submodule R N
    hfin : Module.Finite R (Subtype fun x => Membership.mem M' x)
    left✝ : Module.Finite R (Subtype fun x => Membership.mem w✝ x)
    h : HasSubset.Subset s ↑(LinearMap.range (TensorProduct.mapIncl M' w✝))
    ⊢ Exists fun M' => And (Module.Finite R (Subtype fun x => Membership.mem M' x) …
  -/
  refine ⟨M', hfin, ?_⟩
  /-
    case intro.intro.intro.intro
    R : Type u_1
    M : Type u_2
    N : Type u_3
    inst✝⁴ : CommSemiring R
    inst✝³ : AddCommMonoid M
    inst✝² : AddCommMonoid N
    inst✝¹ : Module R M
    inst✝ : Module R N
    s : Set (TensorProduct R M N)
    hs : s.Finite
    M' : Submodule R M
    w✝ : Submodule R N
    hfin : Module.Finite R (Subtype fun x => Membership.mem M' x)
    left✝ : Module.Finite R (Subtype fun x => Membership.mem w✝ x)
    h : HasSubset.Subset s ↑(LinearMap.range (TensorProduct.mapIncl M' w✝))
    ⊢ HasSubset.Subset s ↑(LinearMap.range (LinearMap.rTensor N M'.subtype))
  -/
  rw [mapIncl, ← LinearMap.rTensor_comp_lTensor] at h
  /-
    case intro.intro.intro.intro
    R : Type u_1
    M : Type u_2
    N : Type u_3
    inst✝⁴ : CommSemiring R
    inst✝³ : AddCommMonoid M
    inst✝² : AddCommMonoid N
    inst✝¹ : Module R M
    inst✝ : Module R N
    s : Set (TensorProduct R M N)
    hs : s.Finite
    M' : Submodule R M
    w✝ : Submodule R N
    hfin : Module.Finite R (Subtype fun x => Membership.mem M' x)
    left✝ : Module.Finite R (Subtype fun x => Membership.mem w✝ x)
    h : HasSubset.Subset s ↑(LinearMap.range ((LinearMap.rTensor N M'.subtype).com …
    ⊢ HasSubset.Subset s ↑(LinearMap.range (LinearMap.rTensor N M'.subtype))
  -/
  exact h.trans (LinearMap.range_comp_le_range _ _)
  /-
    🎉 no goals
  -/


/-- For a finite subset `s` of `M ⊗[R] N`, there exists a finitely generated
submodule `N'` of `N`, such that `s` is contained in the image
of `M ⊗[R] N'` in `M ⊗[R] N`. -/
theorem exists_finite_submodule_right_of_finite (s : Set (M ⊗[R] N)) (hs : s.Finite) :
    ∃ N' : Submodule R N, Module.Finite R N' ∧ s ⊆ LinearMap.range (N'.subtype.lTensor M) := by
  /-
    R : Type u_1
    M : Type u_2
    N : Type u_3
    inst✝⁴ : CommSemiring R
    inst✝³ : AddCommMonoid M
    inst✝² : AddCommMonoid N
    inst✝¹ : Module R M
    inst✝ : Module R N
    s : Set (TensorProduct R M N)
    hs : s.Finite
    ⊢ Exists fun N' => And (Module.Finite R (Subtype fun x => Membership.mem N' x) …
  -/
  obtain ⟨_, N', _, hfin, h⟩ := exists_finite_submodule_of_finite s hs
  /-
    case intro.intro.intro.intro
    R : Type u_1
    M : Type u_2
    N : Type u_3
    inst✝⁴ : CommSemiring R
    inst✝³ : AddCommMonoid M
    inst✝² : AddCommMonoid N
    inst✝¹ : Module R M
    inst✝ : Module R N
    s : Set (TensorProduct R M N)
    hs : s.Finite
    w✝ : Submodule R M
    N' : Submodule R N
    left✝ : Module.Finite R (Subtype fun x => Membership.mem w✝ x)
    hfin : Module.Finite R (Subtype fun x => Membership.mem N' x)
    h : HasSubset.Subset s ↑(LinearMap.range (TensorProduct.mapIncl w✝ N'))
    ⊢ Exists fun N' => And (Module.Finite R (Subtype fun x => Membership.mem N' x) …
  -/
  refine ⟨N', hfin, ?_⟩
  /-
    case intro.intro.intro.intro
    R : Type u_1
    M : Type u_2
    N : Type u_3
    inst✝⁴ : CommSemiring R
    inst✝³ : AddCommMonoid M
    inst✝² : AddCommMonoid N
    inst✝¹ : Module R M
    inst✝ : Module R N
    s : Set (TensorProduct R M N)
    hs : s.Finite
    w✝ : Submodule R M
    N' : Submodule R N
    left✝ : Module.Finite R (Subtype fun x => Membership.mem w✝ x)
    hfin : Module.Finite R (Subtype fun x => Membership.mem N' x)
    h : HasSubset.Subset s ↑(LinearMap.range (TensorProduct.mapIncl w✝ N'))
    ⊢ HasSubset.Subset s ↑(LinearMap.range (LinearMap.lTensor M N'.subtype))
  -/
  rw [mapIncl, ← LinearMap.lTensor_comp_rTensor] at h
  /-
    case intro.intro.intro.intro
    R : Type u_1
    M : Type u_2
    N : Type u_3
    inst✝⁴ : CommSemiring R
    inst✝³ : AddCommMonoid M
    inst✝² : AddCommMonoid N
    inst✝¹ : Module R M
    inst✝ : Module R N
    s : Set (TensorProduct R M N)
    hs : s.Finite
    w✝ : Submodule R M
    N' : Submodule R N
    left✝ : Module.Finite R (Subtype fun x => Membership.mem w✝ x)
    hfin : Module.Finite R (Subtype fun x => Membership.mem N' x)
    h : HasSubset.Subset s ↑(LinearMap.range ((LinearMap.lTensor M N'.subtype).com …
    ⊢ HasSubset.Subset s ↑(LinearMap.range (LinearMap.lTensor M N'.subtype))
  -/
  exact h.trans (LinearMap.range_comp_le_range _ _)
  /-
    🎉 no goals
  -/


/-- Variation of `TensorProduct.exists_finite_submodule_of_finite` where `M` and `N` are
already submodules. -/
theorem exists_finite_submodule_of_finite' (s : Set (M₁ ⊗[R] N₁)) (hs : s.Finite) :
    ∃ (M' : Submodule R M) (N' : Submodule R N) (hM : M' ≤ M₁) (hN : N' ≤ N₁),
      Module.Finite R M' ∧ Module.Finite R N' ∧
        s ⊆ LinearMap.range (TensorProduct.map (inclusion hM) (inclusion hN)) := by
  /-
    R : Type u_1
    M : Type u_2
    N : Type u_3
    inst✝⁴ : CommSemiring R
    inst✝³ : AddCommMonoid M
    inst✝² : AddCommMonoid N
    inst✝¹ : Module R M
    inst✝ : Module R N
    M₁ : Submodule R M
    N₁ : Submodule R N
    s : Set (TensorProduct R (Subtype fun x => Membership.mem M₁ x) (Subtype fun x …
    hs : s.Finite
    ⊢ Exists fun M' => Exists fun N' => Exists fun hM => Exists fun hN => And (Mod …
  -/
  obtain ⟨M', N', _, _, h⟩ := exists_finite_submodule_of_finite s hs
  /-
    case intro.intro.intro.intro
    R : Type u_1
    M : Type u_2
    N : Type u_3
    inst✝⁴ : CommSemiring R
    inst✝³ : AddCommMonoid M
    inst✝² : AddCommMonoid N
    inst✝¹ : Module R M
    inst✝ : Module R N
    M₁ : Submodule R M
    N₁ : Submodule R N
    s : Set (TensorProduct R (Subtype fun x => Membership.mem M₁ x) (Subtype fun x …
    hs : s.Finite
    M' : Submodule R (Subtype fun x => Membership.mem M₁ x)
    N' : Submodule R (Subtype fun x => Membership.mem N₁ x)
    left✝¹ : Module.Finite R (Subtype fun x => Membership.mem M' x)
    left✝ : Module.Finite R (Subtype fun x => Membership.mem N' x)
    h : HasSubset.Subset s ↑(LinearMap.range (TensorProduct.mapIncl M' N'))
    ⊢ Exists fun M' => Exists fun N' => Exists fun hM => Exists fun hN => And (Mod …
  -/
  have hM := map_subtype_le M₁ M'
  /-
    case intro.intro.intro.intro
    R : Type u_1
    M : Type u_2
    N : Type u_3
    inst✝⁴ : CommSemiring R
    inst✝³ : AddCommMonoid M
    inst✝² : AddCommMonoid N
    inst✝¹ : Module R M
    inst✝ : Module R N
    M₁ : Submodule R M
    N₁ : Submodule R N
    s : Set (TensorProduct R (Subtype fun x => Membership.mem M₁ x) (Subtype fun x …
    hs : s.Finite
    M' : Submodule R (Subtype fun x => Membership.mem M₁ x)
    N' : Submodule R (Subtype fun x => Membership.mem N₁ x)
    left✝¹ : Module.Finite R (Subtype fun x => Membership.mem M' x)
    left✝ : Module.Finite R (Subtype fun x => Membership.mem N' x)
    h : HasSubset.Subset s ↑(LinearMap.range (TensorProduct.mapIncl M' N'))
    hM : LE.le (Submodule.map M₁.subtype M') M₁
    ⊢ Exists fun M' => Exists fun N' => Exists fun hM => Exists fun hN => And (Mod …
  -/
  have hN := map_subtype_le N₁ N'
  /-
    case intro.intro.intro.intro
    R : Type u_1
    M : Type u_2
    N : Type u_3
    inst✝⁴ : CommSemiring R
    inst✝³ : AddCommMonoid M
    inst✝² : AddCommMonoid N
    inst✝¹ : Module R M
    inst✝ : Module R N
    M₁ : Submodule R M
    N₁ : Submodule R N
    s : Set (TensorProduct R (Subtype fun x => Membership.mem M₁ x) (Subtype fun x …
    hs : s.Finite
    M' : Submodule R (Subtype fun x => Membership.mem M₁ x)
    N' : Submodule R (Subtype fun x => Membership.mem N₁ x)
    left✝¹ : Module.Finite R (Subtype fun x => Membership.mem M' x)
    left✝ : Module.Finite R (Subtype fun x => Membership.mem N' x)
    h : HasSubset.Subset s ↑(LinearMap.range (TensorProduct.mapIncl M' N'))
    hM : LE.le (Submodule.map M₁.subtype M') M₁
    hN : LE.le (Submodule.map N₁.subtype N') N₁
    ⊢ Exists fun M' => Exists fun N' => Exists fun hM => Exists fun hN => And (Mod …
  -/
  refine ⟨_, _, hM, hN, .map _ _, .map _ _, ?_⟩
  rw [mapIncl,
    show M'.subtype = inclusion hM ∘ₗ M₁.subtype.submoduleMap M' by ext; simp,
    show N'.subtype = inclusion hN ∘ₗ N₁.subtype.submoduleMap N' by ext; simp,
    map_comp] at h
  /-
    case intro.intro.intro.intro
    R : Type u_1
    M : Type u_2
    N : Type u_3
    inst✝⁴ : CommSemiring R
    inst✝³ : AddCommMonoid M
    inst✝² : AddCommMonoid N
    inst✝¹ : Module R M
    inst✝ : Module R N
    M₁ : Submodule R M
    N₁ : Submodule R N
    s : Set (TensorProduct R (Subtype fun x => Membership.mem M₁ x) (Subtype fun x …
    hs : s.Finite
    M' : Submodule R (Subtype fun x => Membership.mem M₁ x)
    N' : Submodule R (Subtype fun x => Membership.mem N₁ x)
    left✝¹ : Module.Finite R (Subtype fun x => Membership.mem M' x)
    left✝ : Module.Finite R (Subtype fun x => Membership.mem N' x)
    hM : LE.le (Submodule.map M₁.subtype M') M₁
    hN : LE.le (Submodule.map N₁.subtype N') N₁
    h : HasSubset.Subset s ↑(LinearMap.range ((TensorProduct.map (Submodule.inclus …
    ⊢ HasSubset.Subset s ↑(LinearMap.range (TensorProduct.map (Submodule.inclusion …
  -/
  exact h.trans (LinearMap.range_comp_le_range _ _)
  /-
    🎉 no goals
  -/


/-- Variation of `TensorProduct.exists_finite_submodule_left_of_finite` where `M` and `N` are
already submodules. -/
theorem exists_finite_submodule_left_of_finite' (s : Set (M₁ ⊗[R] N₁)) (hs : s.Finite) :
    ∃ (M' : Submodule R M) (hM : M' ≤ M₁), Module.Finite R M' ∧
      s ⊆ LinearMap.range ((inclusion hM).rTensor N₁) := by
  /-
    R : Type u_1
    M : Type u_2
    N : Type u_3
    inst✝⁴ : CommSemiring R
    inst✝³ : AddCommMonoid M
    inst✝² : AddCommMonoid N
    inst✝¹ : Module R M
    inst✝ : Module R N
    M₁ : Submodule R M
    N₁ : Submodule R N
    s : Set (TensorProduct R (Subtype fun x => Membership.mem M₁ x) (Subtype fun x …
    hs : s.Finite
    ⊢ Exists fun M' => Exists fun hM => And (Module.Finite R (Subtype fun x => Mem …
  -/
  obtain ⟨M', _, hM, _, hfin, _, h⟩ := exists_finite_submodule_of_finite' s hs
  /-
    case intro.intro.intro.intro.intro.intro
    R : Type u_1
    M : Type u_2
    N : Type u_3
    inst✝⁴ : CommSemiring R
    inst✝³ : AddCommMonoid M
    inst✝² : AddCommMonoid N
    inst✝¹ : Module R M
    inst✝ : Module R N
    M₁ : Submodule R M
    N₁ : Submodule R N
    s : Set (TensorProduct R (Subtype fun x => Membership.mem M₁ x) (Subtype fun x …
    hs : s.Finite
    M' : Submodule R M
    w✝¹ : Submodule R N
    hM : LE.le M' M₁
    w✝ : LE.le w✝¹ N₁
    hfin : Module.Finite R (Subtype fun x => Membership.mem M' x)
    left✝ : Module.Finite R (Subtype fun x => Membership.mem w✝¹ x)
    h : HasSubset.Subset s ↑(LinearMap.range (TensorProduct.map (Submodule.inclusi …
    ⊢ Exists fun M' => Exists fun hM => And (Module.Finite R (Subtype fun x => Mem …
  -/
  refine ⟨M', hM, hfin, ?_⟩
  /-
    case intro.intro.intro.intro.intro.intro
    R : Type u_1
    M : Type u_2
    N : Type u_3
    inst✝⁴ : CommSemiring R
    inst✝³ : AddCommMonoid M
    inst✝² : AddCommMonoid N
    inst✝¹ : Module R M
    inst✝ : Module R N
    M₁ : Submodule R M
    N₁ : Submodule R N
    s : Set (TensorProduct R (Subtype fun x => Membership.mem M₁ x) (Subtype fun x …
    hs : s.Finite
    M' : Submodule R M
    w✝¹ : Submodule R N
    hM : LE.le M' M₁
    w✝ : LE.le w✝¹ N₁
    hfin : Module.Finite R (Subtype fun x => Membership.mem M' x)
    left✝ : Module.Finite R (Subtype fun x => Membership.mem w✝¹ x)
    h : HasSubset.Subset s ↑(LinearMap.range (TensorProduct.map (Submodule.inclusi …
    ⊢ HasSubset.Subset s ↑(LinearMap.range (LinearMap.rTensor (Subtype fun x => Me …
  -/
  rw [← LinearMap.rTensor_comp_lTensor] at h
  /-
    case intro.intro.intro.intro.intro.intro
    R : Type u_1
    M : Type u_2
    N : Type u_3
    inst✝⁴ : CommSemiring R
    inst✝³ : AddCommMonoid M
    inst✝² : AddCommMonoid N
    inst✝¹ : Module R M
    inst✝ : Module R N
    M₁ : Submodule R M
    N₁ : Submodule R N
    s : Set (TensorProduct R (Subtype fun x => Membership.mem M₁ x) (Subtype fun x …
    hs : s.Finite
    M' : Submodule R M
    w✝¹ : Submodule R N
    hM : LE.le M' M₁
    w✝ : LE.le w✝¹ N₁
    hfin : Module.Finite R (Subtype fun x => Membership.mem M' x)
    left✝ : Module.Finite R (Subtype fun x => Membership.mem w✝¹ x)
    h : HasSubset.Subset s ↑(LinearMap.range ((LinearMap.rTensor (Subtype fun x => …
    ⊢ HasSubset.Subset s ↑(LinearMap.range (LinearMap.rTensor (Subtype fun x => Me …
  -/
  exact h.trans (LinearMap.range_comp_le_range _ _)
  /-
    🎉 no goals
  -/


/-- Variation of `TensorProduct.exists_finite_submodule_right_of_finite` where `M` and `N` are
already submodules. -/
theorem exists_finite_submodule_right_of_finite' (s : Set (M₁ ⊗[R] N₁)) (hs : s.Finite) :
    ∃ (N' : Submodule R N) (hN : N' ≤ N₁), Module.Finite R N' ∧
      s ⊆ LinearMap.range ((inclusion hN).lTensor M₁) := by
  /-
    R : Type u_1
    M : Type u_2
    N : Type u_3
    inst✝⁴ : CommSemiring R
    inst✝³ : AddCommMonoid M
    inst✝² : AddCommMonoid N
    inst✝¹ : Module R M
    inst✝ : Module R N
    M₁ : Submodule R M
    N₁ : Submodule R N
    s : Set (TensorProduct R (Subtype fun x => Membership.mem M₁ x) (Subtype fun x …
    hs : s.Finite
    ⊢ Exists fun N' => Exists fun hN => And (Module.Finite R (Subtype fun x => Mem …
  -/
  obtain ⟨_, N', _, hN, _, hfin, h⟩ := exists_finite_submodule_of_finite' s hs
  /-
    case intro.intro.intro.intro.intro.intro
    R : Type u_1
    M : Type u_2
    N : Type u_3
    inst✝⁴ : CommSemiring R
    inst✝³ : AddCommMonoid M
    inst✝² : AddCommMonoid N
    inst✝¹ : Module R M
    inst✝ : Module R N
    M₁ : Submodule R M
    N₁ : Submodule R N
    s : Set (TensorProduct R (Subtype fun x => Membership.mem M₁ x) (Subtype fun x …
    hs : s.Finite
    w✝¹ : Submodule R M
    N' : Submodule R N
    w✝ : LE.le w✝¹ M₁
    hN : LE.le N' N₁
    left✝ : Module.Finite R (Subtype fun x => Membership.mem w✝¹ x)
    hfin : Module.Finite R (Subtype fun x => Membership.mem N' x)
    h : HasSubset.Subset s ↑(LinearMap.range (TensorProduct.map (Submodule.inclusi …
    ⊢ Exists fun N' => Exists fun hN => And (Module.Finite R (Subtype fun x => Mem …
  -/
  refine ⟨N', hN, hfin, ?_⟩
  /-
    case intro.intro.intro.intro.intro.intro
    R : Type u_1
    M : Type u_2
    N : Type u_3
    inst✝⁴ : CommSemiring R
    inst✝³ : AddCommMonoid M
    inst✝² : AddCommMonoid N
    inst✝¹ : Module R M
    inst✝ : Module R N
    M₁ : Submodule R M
    N₁ : Submodule R N
    s : Set (TensorProduct R (Subtype fun x => Membership.mem M₁ x) (Subtype fun x …
    hs : s.Finite
    w✝¹ : Submodule R M
    N' : Submodule R N
    w✝ : LE.le w✝¹ M₁
    hN : LE.le N' N₁
    left✝ : Module.Finite R (Subtype fun x => Membership.mem w✝¹ x)
    hfin : Module.Finite R (Subtype fun x => Membership.mem N' x)
    h : HasSubset.Subset s ↑(LinearMap.range (TensorProduct.map (Submodule.inclusi …
    ⊢ HasSubset.Subset s ↑(LinearMap.range (LinearMap.lTensor (Subtype fun x => Me …
  -/
  rw [← LinearMap.lTensor_comp_rTensor] at h
  /-
    case intro.intro.intro.intro.intro.intro
    R : Type u_1
    M : Type u_2
    N : Type u_3
    inst✝⁴ : CommSemiring R
    inst✝³ : AddCommMonoid M
    inst✝² : AddCommMonoid N
    inst✝¹ : Module R M
    inst✝ : Module R N
    M₁ : Submodule R M
    N₁ : Submodule R N
    s : Set (TensorProduct R (Subtype fun x => Membership.mem M₁ x) (Subtype fun x …
    hs : s.Finite
    w✝¹ : Submodule R M
    N' : Submodule R N
    w✝ : LE.le w✝¹ M₁
    hN : LE.le N' N₁
    left✝ : Module.Finite R (Subtype fun x => Membership.mem w✝¹ x)
    hfin : Module.Finite R (Subtype fun x => Membership.mem N' x)
    h : HasSubset.Subset s ↑(LinearMap.range ((LinearMap.lTensor (Subtype fun x => …
    ⊢ HasSubset.Subset s ↑(LinearMap.range (LinearMap.lTensor (Subtype fun x => Me …
  -/
  exact h.trans (LinearMap.range_comp_le_range _ _)
  /-
    🎉 no goals
  -/


