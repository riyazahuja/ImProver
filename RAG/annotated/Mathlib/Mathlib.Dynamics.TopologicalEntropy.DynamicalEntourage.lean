/-- The dynamical entourage associated to a transformation `T`, entourage `U` and time `n`
is the set of points `(x, y)` such that `(T^[k] x, T^[k] y) ∈ U` for all `k < n`, i.e.
which are `U`-close up to time `n`.-/
def dynEntourage (T : X → X) (U : Set (X × X)) (n : ℕ) : Set (X × X) :=
  ⋂ k < n, (map T T)^[k] ⁻¹' U


lemma dynEntourage_eq_inter_Ico (T : X → X) (U : Set (X × X)) (n : ℕ) :
    dynEntourage T U n = ⋂ k : Ico 0 n, (map T T)^[k] ⁻¹' U := by
  /-
    X : Type u_1
    T : X → X
    U : Set (Prod X X)
    n : Nat
    ⊢ Eq (Dynamics.dynEntourage T U n) (Set.iInter fun k => Set.preimage (Nat.iter …
  -/
  simp [dynEntourage]
  /-
    🎉 no goals
  -/


lemma mem_dynEntourage {T : X → X} {U : Set (X × X)} {n : ℕ} {x y : X} :
    (x, y) ∈ dynEntourage T U n ↔ ∀ k < n, (T^[k] x, T^[k] y) ∈ U := by
  /-
    X : Type u_1
    T : X → X
    U : Set (Prod X X)
    n : Nat
    x y : X
    ⊢ Iff (Membership.mem (Dynamics.dynEntourage T U n) { fst := x, snd := y }) (∀ …
  -/
  simp [dynEntourage]
  /-
    🎉 no goals
  -/


lemma mem_ball_dynEntourage {T : X → X} {U : Set (X × X)} {n : ℕ} {x y : X} :
    y ∈ ball x (dynEntourage T U n) ↔ ∀ k < n, T^[k] y ∈ ball (T^[k] x) U := by
  /-
    X : Type u_1
    T : X → X
    U : Set (Prod X X)
    n : Nat
    x y : X
    ⊢ Iff (Membership.mem (UniformSpace.ball x (Dynamics.dynEntourage T U n)) y) ( …
  -/
  simp only [ball, mem_preimage]; exact mem_dynEntourage
                                  /-
                                    🎉 no goals
                                  -/


lemma dynEntourage_mem_uniformity [UniformSpace X] {T : X → X} (h : UniformContinuous T)
    {U : Set (X × X)} (U_uni : U ∈ 𝓤 X) (n : ℕ) :
    dynEntourage T U n ∈ 𝓤 X := by
  /-
    X : Type u_1
    inst✝ : UniformSpace X
    T : X → X
    h : UniformContinuous T
    U : Set (Prod X X)
    U_uni : Membership.mem (uniformity X) U
    n : Nat
    ⊢ Membership.mem (uniformity X) (Dynamics.dynEntourage T U n)
  -/
  rw [dynEntourage_eq_inter_Ico T U n]
  /-
    X : Type u_1
    inst✝ : UniformSpace X
    T : X → X
    h : UniformContinuous T
    U : Set (Prod X X)
    U_uni : Membership.mem (uniformity X) U
    n : Nat
    ⊢ Membership.mem (uniformity X) (Set.iInter fun k => Set.preimage (Nat.iterate …
  -/
  refine Filter.iInter_mem.2 fun k ↦ ?_
  /-
    X : Type u_1
    inst✝ : UniformSpace X
    T : X → X
    h : UniformContinuous T
    U : Set (Prod X X)
    U_uni : Membership.mem (uniformity X) U
    n : Nat
    k : ↑(Set.Ico 0 n)
    ⊢ Membership.mem (uniformity X) (Set.preimage (Nat.iterate (Prod.map T T) ↑k) U)
  -/
  rw [map_iterate T T k]
  /-
    X : Type u_1
    inst✝ : UniformSpace X
    T : X → X
    h : UniformContinuous T
    U : Set (Prod X X)
    U_uni : Membership.mem (uniformity X) U
    n : Nat
    k : ↑(Set.Ico 0 n)
    ⊢ Membership.mem (uniformity X) (Set.preimage (Prod.map (Nat.iterate T ↑k) (Na …
  -/
  exact uniformContinuous_def.1 (UniformContinuous.iterate T k h) U U_uni
  /-
    🎉 no goals
  -/


lemma idRel_subset_dynEntourage (T : X → X) {U : Set (X × X)} (h : idRel ⊆ U) (n : ℕ) :
    idRel ⊆ (dynEntourage T U n) := by
  /-
    X : Type u_1
    T : X → X
    U : Set (Prod X X)
    h : HasSubset.Subset idRel U
    n : Nat
    ⊢ HasSubset.Subset idRel (Dynamics.dynEntourage T U n)
  -/
  simp only [dynEntourage, map_iterate, subset_iInter_iff, idRel_subset, mem_preimage, map_apply]
  /-
    X : Type u_1
    T : X → X
    U : Set (Prod X X)
    h : HasSubset.Subset idRel U
    n : Nat
    ⊢ ∀ (i : Nat), LT.lt i n → ∀ (a : X), Membership.mem U { fst := Nat.iterate T  …
  -/
  exact fun _ _ _ ↦ h rfl
  /-
    🎉 no goals
  -/


lemma _root_.SymmetricRel.dynEntourage (T : X → X) {U : Set (X × X)} (h : SymmetricRel U) (n : ℕ) :
    SymmetricRel (dynEntourage T U n) := by
  /-
    X : Type u_1
    T : X → X
    U : Set (Prod X X)
    h : SymmetricRel U
    n : Nat
    ⊢ SymmetricRel (Dynamics.dynEntourage T U n)
  -/
  ext xy
  /-
    case h
    X : Type u_1
    T : X → X
    U : Set (Prod X X)
    h : SymmetricRel U
    n : Nat
    xy : Prod X X
    ⊢ Iff (Membership.mem (Set.preimage Prod.swap (Dynamics.dynEntourage T U n)) x …
  -/
  simp only [Dynamics.dynEntourage, map_iterate, mem_preimage, mem_iInter]
  /-
    case h
    X : Type u_1
    T : X → X
    U : Set (Prod X X)
    h : SymmetricRel U
    n : Nat
    xy : Prod X X
    ⊢ Iff (∀ (i : Nat), LT.lt i n → Membership.mem U (Prod.map (Nat.iterate T i) ( …
  -/
  refine forall₂_congr fun k _ ↦ ?_
  /-
    case h
    X : Type u_1
    T : X → X
    U : Set (Prod X X)
    h : SymmetricRel U
    n : Nat
    xy : Prod X X
    k : Nat
    x✝ : LT.lt k n
    ⊢ Iff (Membership.mem U (Prod.map (Nat.iterate T k) (Nat.iterate T k) xy.swap) …
  -/
  exact map_apply' _ _ _ ▸ SymmetricRel.mk_mem_comm h
  /-
    🎉 no goals
  -/


lemma dynEntourage_comp_subset (T : X → X) (U V : Set (X × X)) (n : ℕ) :
    (dynEntourage T U n) ○ (dynEntourage T V n) ⊆ dynEntourage T (U ○ V) n := by
  /-
    X : Type u_1
    T : X → X
    U V : Set (Prod X X)
    n : Nat
    ⊢ HasSubset.Subset (compRel (Dynamics.dynEntourage T U n) (Dynamics.dynEntoura …
  -/
  simp only [dynEntourage, map_iterate, subset_iInter_iff]
  /-
    X : Type u_1
    T : X → X
    U V : Set (Prod X X)
    n : Nat
    ⊢ ∀ (i : Nat), LT.lt i n → HasSubset.Subset (compRel (Set.iInter fun k => Set. …
  -/
  intro k k_n xy xy_comp
  /-
    X : Type u_1
    T : X → X
    U V : Set (Prod X X)
    n k : Nat
    k_n : LT.lt k n
    xy : Prod X X
    xy_comp : Membership.mem (compRel (Set.iInter fun k => Set.iInter fun x => Set …
    ⊢ Membership.mem (Set.preimage (Prod.map (Nat.iterate T k) (Nat.iterate T k))  …
  -/
  simp only [compRel, mem_iInter, mem_preimage, map_apply, mem_setOf_eq] at xy_comp ⊢
  /-
    X : Type u_1
    T : X → X
    U V : Set (Prod X X)
    n k : Nat
    k_n : LT.lt k n
    xy : Prod X X
    xy_comp : Exists fun z => And (∀ (i : Nat), LT.lt i n → Membership.mem U { fst …
    ⊢ Exists fun z => And (Membership.mem U { fst := (Prod.map (Nat.iterate T k) ( …
  -/
  rcases xy_comp with ⟨z, hz1, hz2⟩
  /-
    case intro.intro
    X : Type u_1
    T : X → X
    U V : Set (Prod X X)
    n k : Nat
    k_n : LT.lt k n
    xy : Prod X X
    z : X
    hz1 : ∀ (i : Nat), LT.lt i n → Membership.mem U { fst := Nat.iterate T i xy.1, …
    hz2 : ∀ (i : Nat), LT.lt i n → Membership.mem V { fst := Nat.iterate T i z, sn …
    ⊢ Exists fun z => And (Membership.mem U { fst := (Prod.map (Nat.iterate T k) ( …
  -/
  exact mem_ball_comp (hz1 k k_n) (hz2 k k_n)
  /-
    🎉 no goals
  -/


lemma _root_.isOpen.dynEntourage [TopologicalSpace X] {T : X → X} (T_cont : Continuous T)
    {U : Set (X × X)} (U_open : IsOpen U) (n : ℕ) :
    IsOpen (dynEntourage T U n) := by
  /-
    X : Type u_1
    inst✝ : TopologicalSpace X
    T : X → X
    T_cont : Continuous T
    U : Set (Prod X X)
    U_open : IsOpen U
    n : Nat
    ⊢ IsOpen (Dynamics.dynEntourage T U n)
  -/
  rw [dynEntourage_eq_inter_Ico T U n]
  /-
    X : Type u_1
    inst✝ : TopologicalSpace X
    T : X → X
    T_cont : Continuous T
    U : Set (Prod X X)
    U_open : IsOpen U
    n : Nat
    ⊢ IsOpen (Set.iInter fun k => Set.preimage (Nat.iterate (Prod.map T T) ↑k) U)
  -/
  refine isOpen_iInter_of_finite fun k ↦ ?_
  /-
    X : Type u_1
    inst✝ : TopologicalSpace X
    T : X → X
    T_cont : Continuous T
    U : Set (Prod X X)
    U_open : IsOpen U
    n : Nat
    k : ↑(Set.Ico 0 n)
    ⊢ IsOpen (Set.preimage (Nat.iterate (Prod.map T T) ↑k) U)
  -/
  exact U_open.preimage ((T_cont.prodMap T_cont).iterate k)
  /-
    🎉 no goals
  -/


lemma dynEntourage_monotone (T : X → X) (n : ℕ) :
    Monotone (fun U : Set (X × X) ↦ dynEntourage T U n) :=
  fun _ _ h ↦ iInter₂_mono fun _ _ ↦ preimage_mono h


lemma dynEntourage_antitone (T : X → X) (U : Set (X × X)) :
    Antitone (fun n : ℕ ↦ dynEntourage T U n) :=
                                             /-
                                               X : Type u_1
                                               T : X → X
                                               U : Set (Prod X X)
                                               m n : Nat
                                               m_n : LE.le m n
                                               k : Nat
                                               k_m : LT.lt k m
                                               ⊢ Exists fun i => Exists fun j => HasSubset.Subset (Set.preimage (Nat.iterate  …
                                             -/
  fun m n m_n ↦ iInter₂_mono' fun k k_m ↦ by use k, lt_of_lt_of_le k_m m_n
                                             /-
                                               🎉 no goals
                                             -/


@[simp]
lemma dynEntourage_zero {T : X → X} {U : Set (X × X)} :
                                    /-
                                      X : Type u_1
                                      T : X → X
                                      U : Set (Prod X X)
                                      ⊢ Eq (Dynamics.dynEntourage T U 0) Set.univ
                                    -/
    dynEntourage T U 0 = univ := by simp [dynEntourage]
                                    /-
                                      🎉 no goals
                                    -/


@[simp]
lemma dynEntourage_one {T : X → X} {U : Set (X × X)} :
                                 /-
                                   X : Type u_1
                                   T : X → X
                                   U : Set (Prod X X)
                                   ⊢ Eq (Dynamics.dynEntourage T U 1) U
                                 -/
    dynEntourage T U 1 = U := by simp [dynEntourage]
                                 /-
                                   🎉 no goals
                                 -/


@[simp]
lemma dynEntourage_univ {T : X → X} {n : ℕ} :
                                       /-
                                         X : Type u_1
                                         T : X → X
                                         n : Nat
                                         ⊢ Eq (Dynamics.dynEntourage T Set.univ n) Set.univ
                                       -/
    dynEntourage T univ n = univ := by simp [dynEntourage]
                                       /-
                                         🎉 no goals
                                       -/


lemma mem_ball_dynEntourage_comp (T : X → X) (n : ℕ) {U : Set (X × X)} (U_symm : SymmetricRel U)
    (x y : X) (h : (ball x (dynEntourage T U n) ∩ ball y (dynEntourage T U n)).Nonempty) :
    x ∈ ball y (dynEntourage T (U ○ U) n) := by
  /-
    X : Type u_1
    T : X → X
    n : Nat
    U : Set (Prod X X)
    U_symm : SymmetricRel U
    x y : X
    h : (Inter.inter (UniformSpace.ball x (Dynamics.dynEntourage T U n)) (UniformS …
    ⊢ Membership.mem (UniformSpace.ball y (Dynamics.dynEntourage T (compRel U U) n …
  -/
  rcases h with ⟨z, z_Bx, z_By⟩
  /-
    case intro.intro
    X : Type u_1
    T : X → X
    n : Nat
    U : Set (Prod X X)
    U_symm : SymmetricRel U
    x y z : X
    z_Bx : Membership.mem (UniformSpace.ball x (Dynamics.dynEntourage T U n)) z
    z_By : Membership.mem (UniformSpace.ball y (Dynamics.dynEntourage T U n)) z
    ⊢ Membership.mem (UniformSpace.ball y (Dynamics.dynEntourage T (compRel U U) n …
  -/
  rw [mem_ball_symmetry (SymmetricRel.dynEntourage T U_symm n)] at z_Bx
  /-
    case intro.intro
    X : Type u_1
    T : X → X
    n : Nat
    U : Set (Prod X X)
    U_symm : SymmetricRel U
    x y z : X
    z_Bx : Membership.mem (UniformSpace.ball z (Dynamics.dynEntourage T U n)) x
    z_By : Membership.mem (UniformSpace.ball y (Dynamics.dynEntourage T U n)) z
    ⊢ Membership.mem (UniformSpace.ball y (Dynamics.dynEntourage T (compRel U U) n …
  -/
  exact dynEntourage_comp_subset T U U n (mem_ball_comp z_By z_Bx)
  /-
    🎉 no goals
  -/


lemma _root_.Function.Semiconj.preimage_dynEntourage {Y : Type*} {S : X → X} {T : Y → Y} {φ : X → Y}
    (h : Function.Semiconj φ S T) (U : Set (Y × Y)) (n : ℕ) :
    (map φ φ)⁻¹' (dynEntourage T U n) = dynEntourage S ((map φ φ)⁻¹' U) n := by
  /-
    X : Type u_1
    Y : Type u_2
    S : X → X
    T : Y → Y
    φ : X → Y
    h : Function.Semiconj φ S T
    U : Set (Prod Y Y)
    n : Nat
    ⊢ Eq (Set.preimage (Prod.map φ φ) (Dynamics.dynEntourage T U n)) (Dynamics.dyn …
  -/
  rw [dynEntourage, preimage_iInter₂]
  /-
    X : Type u_1
    Y : Type u_2
    S : X → X
    T : Y → Y
    φ : X → Y
    h : Function.Semiconj φ S T
    U : Set (Prod Y Y)
    n : Nat
    ⊢ Eq (Set.iInter fun i => Set.iInter fun j => Set.preimage (Prod.map φ φ) (Set …
  -/
  refine iInter₂_congr fun k _ ↦ ?_
  rw [← preimage_comp, ← preimage_comp, map_iterate S S k, map_iterate T T k, map_comp_map,
    map_comp_map, (Function.Semiconj.iterate_right h k).comp_eq]


