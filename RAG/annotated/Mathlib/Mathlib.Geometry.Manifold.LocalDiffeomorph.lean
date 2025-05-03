/-- A partial diffeomorphism on `s` is a function `f : M → N` such that `f` restricts to a
diffeomorphism `s → t` between open subsets of `M` and `N`, respectively.
This is an auxiliary definition and should not be used outside of this file. -/
structure PartialDiffeomorph extends PartialEquiv M N where
  open_source : IsOpen source
  open_target : IsOpen target
  contMDiffOn_toFun : ContMDiffOn I J n toFun source
  contMDiffOn_invFun : ContMDiffOn J I n invFun target


/-- Coercion of a `PartialDiffeomorph` to function.
Note that a `PartialDiffeomorph` is not `DFunLike` (like `PartialHomeomorph`),
as `toFun` doesn't determine `invFun` outside of `target`. -/
instance : CoeFun (PartialDiffeomorph I J M N n) fun _ => M → N :=
  ⟨fun Φ => Φ.toFun⟩


/-- A diffeomorphism is a partial diffeomorphism. -/
def Diffeomorph.toPartialDiffeomorph (h : Diffeomorph I J M N n) :
    PartialDiffeomorph I J M N n where
  toPartialEquiv := h.toHomeomorph.toPartialEquiv
  open_source := isOpen_univ
  open_target := isOpen_univ
  contMDiffOn_toFun x _ := h.contMDiff_toFun x
  contMDiffOn_invFun _ _ := h.symm.contMDiffWithinAt

-- Add the very basic API we need.

/-- A partial diffeomorphism is also a local homeomorphism. -/
def toPartialHomeomorph : PartialHomeomorph M N where
  toPartialEquiv := Φ.toPartialEquiv
  open_source := Φ.open_source
  open_target := Φ.open_target
  continuousOn_toFun := Φ.contMDiffOn_toFun.continuousOn
  continuousOn_invFun := Φ.contMDiffOn_invFun.continuousOn


/-- The inverse of a local diffeomorphism. -/
protected def symm : PartialDiffeomorph J I N M n where
  toPartialEquiv := Φ.toPartialEquiv.symm
  open_source := Φ.open_target
  open_target := Φ.open_source
  contMDiffOn_toFun := Φ.contMDiffOn_invFun
  contMDiffOn_invFun := Φ.contMDiffOn_toFun


protected theorem contMDiffOn : ContMDiffOn I J n Φ Φ.source :=
  Φ.contMDiffOn_toFun


protected theorem mdifferentiableOn (hn : 1 ≤ n) : MDifferentiableOn I J Φ Φ.source :=
  (Φ.contMDiffOn).mdifferentiableOn hn


protected theorem mdifferentiableAt (hn : 1 ≤ n) {x : M} (hx : x ∈ Φ.source) :
    MDifferentiableAt I J Φ x :=
  (Φ.mdifferentiableOn hn x hx).mdifferentiableAt (Φ.open_source.mem_nhds hx)

/- We could add lots of additional API (following `Diffeomorph` and `PartialHomeomorph`), such as
* further continuity and differentiability lemmas
* refl and trans instances; lemmas between them.
As this declaration is meant for internal use only, we keep it simple. -/

/-- `f : M → N` is called a **`C^n` local diffeomorphism at *x*** iff there exist
  open sets `U ∋ x` and `V ∋ f x` and a diffeomorphism `Φ : U → V` such that `f = Φ` on `U`. -/
def IsLocalDiffeomorphAt (f : M → N) (x : M) : Prop :=
  ∃ Φ : PartialDiffeomorph I J M N n, x ∈ Φ.source ∧ EqOn f Φ Φ.source


/-- `f : M → N` is called a **`C^n` local diffeomorphism on *s*** iff it is a local diffeomorphism
  at each `x : s`. -/
def IsLocalDiffeomorphOn (f : M → N) (s : Set M) : Prop :=
  ∀ x : s, IsLocalDiffeomorphAt I J n f x


/-- `f : M → N` is a **`C^n` local diffeomorphism** iff it is a local diffeomorphism
at each `x ∈ M`. -/
def IsLocalDiffeomorph (f : M → N) : Prop :=
  ∀ x : M, IsLocalDiffeomorphAt I J n f x


variable {I J n} in
lemma isLocalDiffeomorphOn_iff {f : M → N} (s : Set M) :
                                                                                   /-
                                                                                     𝕜 : Type u_1
                                                                                     inst✝¹⁰ : NontriviallyNormedField 𝕜
                                                                                     E : Type u_2
                                                                                     inst✝⁹ : NormedAddCommGroup E
                                                                                     inst✝⁸ : NormedSpace 𝕜 E
                                                                                     F : Type u_3
                                                                                     inst✝⁷ : NormedAddCommGroup F
                                                                                     inst✝⁶ : NormedSpace 𝕜 F
                                                                                     H : Type u_4
                                                                                     inst✝⁵ : TopologicalSpace H
                                                                                     G : Type u_5
                                                                                     inst✝⁴ : TopologicalSpace G
                                                                                     I : ModelWithCorners 𝕜 E H
                                                                                     J : ModelWithCorners 𝕜 F G
                                                                                     M : Type u_6
                                                                                     inst✝³ : TopologicalSpace M
                                                                                     inst✝² : ChartedSpace H M
                                                                                     N : Type u_7
                                                                                     inst✝¹ : TopologicalSpace N
                                                                                     inst✝ : ChartedSpace G N
                                                                                     n : ENat
                                                                                     f : M → N
                                                                                     s : Set M
                                                                                     ⊢ Iff (IsLocalDiffeomorphOn I J n f s) (∀ (x : ↑s), IsLocalDiffeomorphAt I J n …
                                                                                   -/
    IsLocalDiffeomorphOn I J n f s ↔ ∀ x : s, IsLocalDiffeomorphAt I J n f x := by rfl
                                                                                   /-
                                                                                     🎉 no goals
                                                                                   -/


variable {I J n} in
lemma isLocalDiffeomorph_iff {f : M → N} :
                                                                               /-
                                                                                 𝕜 : Type u_1
                                                                                 inst✝¹⁰ : NontriviallyNormedField 𝕜
                                                                                 E : Type u_2
                                                                                 inst✝⁹ : NormedAddCommGroup E
                                                                                 inst✝⁸ : NormedSpace 𝕜 E
                                                                                 F : Type u_3
                                                                                 inst✝⁷ : NormedAddCommGroup F
                                                                                 inst✝⁶ : NormedSpace 𝕜 F
                                                                                 H : Type u_4
                                                                                 inst✝⁵ : TopologicalSpace H
                                                                                 G : Type u_5
                                                                                 inst✝⁴ : TopologicalSpace G
                                                                                 I : ModelWithCorners 𝕜 E H
                                                                                 J : ModelWithCorners 𝕜 F G
                                                                                 M : Type u_6
                                                                                 inst✝³ : TopologicalSpace M
                                                                                 inst✝² : ChartedSpace H M
                                                                                 N : Type u_7
                                                                                 inst✝¹ : TopologicalSpace N
                                                                                 inst✝ : ChartedSpace G N
                                                                                 n : ENat
                                                                                 f : M → N
                                                                                 ⊢ Iff (IsLocalDiffeomorph I J n f) (∀ (x : M), IsLocalDiffeomorphAt I J n f x)
                                                                               -/
    IsLocalDiffeomorph I J n f ↔ ∀ x : M, IsLocalDiffeomorphAt I J n f x := by rfl
                                                                               /-
                                                                                 🎉 no goals
                                                                               -/


variable {I J n} in
theorem isLocalDiffeomorph_iff_isLocalDiffeomorphOn_univ {f : M → N} :
    IsLocalDiffeomorph I J n f ↔ IsLocalDiffeomorphOn I J n f Set.univ :=
  ⟨fun hf x ↦ hf x, fun hf x ↦ hf ⟨x, trivial⟩⟩


variable {I J n} in
lemma IsLocalDiffeomorph.isLocalDiffeomorphOn
    {f : M → N} (hf : IsLocalDiffeomorph I J n f) (s : Set M) : IsLocalDiffeomorphOn I J n f s :=
  fun x ↦ hf x


/-- A `C^n` local diffeomorphism at `x` is `C^n` differentiable at `x`. -/
lemma IsLocalDiffeomorphAt.contMDiffAt (hf : IsLocalDiffeomorphAt I J n f x) :
    ContMDiffAt I J n f x := by
  /-
    𝕜 : Type u_1
    inst✝¹⁰ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁹ : NormedAddCommGroup E
    inst✝⁸ : NormedSpace 𝕜 E
    F : Type u_3
    inst✝⁷ : NormedAddCommGroup F
    inst✝⁶ : NormedSpace 𝕜 F
    H : Type u_4
    inst✝⁵ : TopologicalSpace H
    G : Type u_5
    inst✝⁴ : TopologicalSpace G
    I : ModelWithCorners 𝕜 E H
    J : ModelWithCorners 𝕜 F G
    M : Type u_6
    inst✝³ : TopologicalSpace M
    inst✝² : ChartedSpace H M
    N : Type u_7
    inst✝¹ : TopologicalSpace N
    inst✝ : ChartedSpace G N
    n : ENat
    f : M → N
    x : M
    hf : IsLocalDiffeomorphAt I J n f x
    ⊢ ContMDiffAt I J n f x
  -/
  choose Φ hx heq using hf
  -- In fact, even `ContMDiffOn I J n f Φ.source`.
  /-
    𝕜 : Type u_1
    inst✝¹⁰ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁹ : NormedAddCommGroup E
    inst✝⁸ : NormedSpace 𝕜 E
    F : Type u_3
    inst✝⁷ : NormedAddCommGroup F
    inst✝⁶ : NormedSpace 𝕜 F
    H : Type u_4
    inst✝⁵ : TopologicalSpace H
    G : Type u_5
    inst✝⁴ : TopologicalSpace G
    I : ModelWithCorners 𝕜 E H
    J : ModelWithCorners 𝕜 F G
    M : Type u_6
    inst✝³ : TopologicalSpace M
    inst✝² : ChartedSpace H M
    N : Type u_7
    inst✝¹ : TopologicalSpace N
    inst✝ : ChartedSpace G N
    n : ENat
    f : M → N
    x : M
    Φ : PartialDiffeomorph I J M N n
    hx : Membership.mem Φ.source x
    heq : Set.EqOn f (↑Φ.toPartialEquiv) Φ.source
    ⊢ ContMDiffAt I J n f x
  -/
  exact ((Φ.contMDiffOn_toFun).congr heq).contMDiffAt (Φ.open_source.mem_nhds hx)
  /-
    🎉 no goals
  -/


/-- A local diffeomorphism at `x` is differentiable at `x`. -/
lemma IsLocalDiffeomorphAt.mdifferentiableAt (hf : IsLocalDiffeomorphAt I J n f x) (hn : 1 ≤ n) :
    MDifferentiableAt I J f x :=
  hf.contMDiffAt.mdifferentiableAt hn


/-- A `C^n` local diffeomorphism on `s` is `C^n` on `s`. -/
lemma IsLocalDiffeomorphOn.contMDiffOn (hf : IsLocalDiffeomorphOn I J n f s) :
    ContMDiffOn I J n f s :=
  fun x hx ↦ (hf ⟨x, hx⟩).contMDiffAt.contMDiffWithinAt


/-- A local diffeomorphism on `s` is differentiable on `s`. -/
lemma IsLocalDiffeomorphOn.mdifferentiableOn (hf : IsLocalDiffeomorphOn I J n f s) (hn : 1 ≤ n) :
    MDifferentiableOn I J f s :=
  hf.contMDiffOn.mdifferentiableOn hn


/-- A `C^n` local diffeomorphism is `C^n`. -/
lemma IsLocalDiffeomorph.contMDiff (hf : IsLocalDiffeomorph I J n f) : ContMDiff I J n f :=
  fun x ↦ (hf x).contMDiffAt


/-- A `C^n` local diffeomorphism is differentiable. -/
lemma IsLocalDiffeomorph.mdifferentiable (hf : IsLocalDiffeomorph I J n f) (hn : 1 ≤ n) :
    MDifferentiable I J f :=
  fun x ↦ (hf x).mdifferentiableAt hn


/-- A `C^n` diffeomorphism is a local diffeomorphism. -/
lemma Diffeomorph.isLocalDiffeomorph (Φ : M ≃ₘ^n⟮I, J⟯ N) : IsLocalDiffeomorph I J n Φ :=
                                       /-
                                         𝕜 : Type u_1
                                         inst✝¹⁰ : NontriviallyNormedField 𝕜
                                         E : Type u_2
                                         inst✝⁹ : NormedAddCommGroup E
                                         inst✝⁸ : NormedSpace 𝕜 E
                                         F : Type u_3
                                         inst✝⁷ : NormedAddCommGroup F
                                         inst✝⁶ : NormedSpace 𝕜 F
                                         H : Type u_4
                                         inst✝⁵ : TopologicalSpace H
                                         G : Type u_5
                                         inst✝⁴ : TopologicalSpace G
                                         I : ModelWithCorners 𝕜 E H
                                         J : ModelWithCorners 𝕜 F G
                                         M : Type u_6
                                         inst✝³ : TopologicalSpace M
                                         inst✝² : ChartedSpace H M
                                         N : Type u_7
                                         inst✝¹ : TopologicalSpace N
                                         inst✝ : ChartedSpace G N
                                         n : ENat
                                         Φ : Diffeomorph I J M N n
                                         _x : M
                                         ⊢ Membership.mem Φ.toPartialDiffeomorph.source _x
                                       -/
  fun _x ↦ ⟨Φ.toPartialDiffeomorph, by trivial, eqOn_refl Φ _⟩
                                       /-
                                         🎉 no goals
                                       -/

-- FUTURE: if useful, also add "a `PartialDiffeomorph` is a local diffeomorphism on its source"


/-- A local diffeomorphism on `s` is a local homeomorphism on `s`. -/
theorem IsLocalDiffeomorphOn.isLocalHomeomorphOn {s : Set M} (hf : IsLocalDiffeomorphOn I J n f s) :
    IsLocalHomeomorphOn f s := by
  /-
    𝕜 : Type u_1
    inst✝¹⁰ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁹ : NormedAddCommGroup E
    inst✝⁸ : NormedSpace 𝕜 E
    F : Type u_3
    inst✝⁷ : NormedAddCommGroup F
    inst✝⁶ : NormedSpace 𝕜 F
    H : Type u_4
    inst✝⁵ : TopologicalSpace H
    G : Type u_5
    inst✝⁴ : TopologicalSpace G
    I : ModelWithCorners 𝕜 E H
    J : ModelWithCorners 𝕜 F G
    M : Type u_6
    inst✝³ : TopologicalSpace M
    inst✝² : ChartedSpace H M
    N : Type u_7
    inst✝¹ : TopologicalSpace N
    inst✝ : ChartedSpace G N
    n : ENat
    f : M → N
    s : Set M
    hf : IsLocalDiffeomorphOn I J n f s
    ⊢ IsLocalHomeomorphOn f s
  -/
  apply IsLocalHomeomorphOn.mk
  /-
    case h
    𝕜 : Type u_1
    inst✝¹⁰ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁹ : NormedAddCommGroup E
    inst✝⁸ : NormedSpace 𝕜 E
    F : Type u_3
    inst✝⁷ : NormedAddCommGroup F
    inst✝⁶ : NormedSpace 𝕜 F
    H : Type u_4
    inst✝⁵ : TopologicalSpace H
    G : Type u_5
    inst✝⁴ : TopologicalSpace G
    I : ModelWithCorners 𝕜 E H
    J : ModelWithCorners 𝕜 F G
    M : Type u_6
    inst✝³ : TopologicalSpace M
    inst✝² : ChartedSpace H M
    N : Type u_7
    inst✝¹ : TopologicalSpace N
    inst✝ : ChartedSpace G N
    n : ENat
    f : M → N
    s : Set M
    hf : IsLocalDiffeomorphOn I J n f s
    ⊢ ∀ (x : M), Membership.mem s x → Exists fun e => And (Membership.mem e.source …
  -/
  intro x hx
  /-
    case h
    𝕜 : Type u_1
    inst✝¹⁰ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁹ : NormedAddCommGroup E
    inst✝⁸ : NormedSpace 𝕜 E
    F : Type u_3
    inst✝⁷ : NormedAddCommGroup F
    inst✝⁶ : NormedSpace 𝕜 F
    H : Type u_4
    inst✝⁵ : TopologicalSpace H
    G : Type u_5
    inst✝⁴ : TopologicalSpace G
    I : ModelWithCorners 𝕜 E H
    J : ModelWithCorners 𝕜 F G
    M : Type u_6
    inst✝³ : TopologicalSpace M
    inst✝² : ChartedSpace H M
    N : Type u_7
    inst✝¹ : TopologicalSpace N
    inst✝ : ChartedSpace G N
    n : ENat
    f : M → N
    s : Set M
    hf : IsLocalDiffeomorphOn I J n f s
    x : M
    hx : Membership.mem s x
    ⊢ Exists fun e => And (Membership.mem e.source x) (Set.EqOn f (↑e) e.source)
  -/
  choose U hyp using hf ⟨x, hx⟩
  /-
    case h
    𝕜 : Type u_1
    inst✝¹⁰ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁹ : NormedAddCommGroup E
    inst✝⁸ : NormedSpace 𝕜 E
    F : Type u_3
    inst✝⁷ : NormedAddCommGroup F
    inst✝⁶ : NormedSpace 𝕜 F
    H : Type u_4
    inst✝⁵ : TopologicalSpace H
    G : Type u_5
    inst✝⁴ : TopologicalSpace G
    I : ModelWithCorners 𝕜 E H
    J : ModelWithCorners 𝕜 F G
    M : Type u_6
    inst✝³ : TopologicalSpace M
    inst✝² : ChartedSpace H M
    N : Type u_7
    inst✝¹ : TopologicalSpace N
    inst✝ : ChartedSpace G N
    n : ENat
    f : M → N
    s : Set M
    hf : IsLocalDiffeomorphOn I J n f s
    x : M
    hx : Membership.mem s x
    U : PartialDiffeomorph I J M N n
    hyp : And (Membership.mem U.source ↑⟨x, hx⟩) (Set.EqOn f (↑U.toPartialEquiv) U …
    ⊢ Exists fun e => And (Membership.mem e.source x) (Set.EqOn f (↑e) e.source)
  -/
  exact ⟨U.toPartialHomeomorph, hyp⟩
  /-
    🎉 no goals
  -/


/-- A local diffeomorphism is a local homeomorphism. -/
theorem IsLocalDiffeomorph.isLocalHomeomorph (hf : IsLocalDiffeomorph I J n f) :
    IsLocalHomeomorph f := by
  /-
    𝕜 : Type u_1
    inst✝¹⁰ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁹ : NormedAddCommGroup E
    inst✝⁸ : NormedSpace 𝕜 E
    F : Type u_3
    inst✝⁷ : NormedAddCommGroup F
    inst✝⁶ : NormedSpace 𝕜 F
    H : Type u_4
    inst✝⁵ : TopologicalSpace H
    G : Type u_5
    inst✝⁴ : TopologicalSpace G
    I : ModelWithCorners 𝕜 E H
    J : ModelWithCorners 𝕜 F G
    M : Type u_6
    inst✝³ : TopologicalSpace M
    inst✝² : ChartedSpace H M
    N : Type u_7
    inst✝¹ : TopologicalSpace N
    inst✝ : ChartedSpace G N
    n : ENat
    f : M → N
    hf : IsLocalDiffeomorph I J n f
    ⊢ IsLocalHomeomorph f
  -/
  rw [isLocalHomeomorph_iff_isLocalHomeomorphOn_univ]
  /-
    𝕜 : Type u_1
    inst✝¹⁰ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁹ : NormedAddCommGroup E
    inst✝⁸ : NormedSpace 𝕜 E
    F : Type u_3
    inst✝⁷ : NormedAddCommGroup F
    inst✝⁶ : NormedSpace 𝕜 F
    H : Type u_4
    inst✝⁵ : TopologicalSpace H
    G : Type u_5
    inst✝⁴ : TopologicalSpace G
    I : ModelWithCorners 𝕜 E H
    J : ModelWithCorners 𝕜 F G
    M : Type u_6
    inst✝³ : TopologicalSpace M
    inst✝² : ChartedSpace H M
    N : Type u_7
    inst✝¹ : TopologicalSpace N
    inst✝ : ChartedSpace G N
    n : ENat
    f : M → N
    hf : IsLocalDiffeomorph I J n f
    ⊢ IsLocalHomeomorphOn f Set.univ
  -/
  rw [isLocalDiffeomorph_iff_isLocalDiffeomorphOn_univ] at hf
  /-
    𝕜 : Type u_1
    inst✝¹⁰ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁹ : NormedAddCommGroup E
    inst✝⁸ : NormedSpace 𝕜 E
    F : Type u_3
    inst✝⁷ : NormedAddCommGroup F
    inst✝⁶ : NormedSpace 𝕜 F
    H : Type u_4
    inst✝⁵ : TopologicalSpace H
    G : Type u_5
    inst✝⁴ : TopologicalSpace G
    I : ModelWithCorners 𝕜 E H
    J : ModelWithCorners 𝕜 F G
    M : Type u_6
    inst✝³ : TopologicalSpace M
    inst✝² : ChartedSpace H M
    N : Type u_7
    inst✝¹ : TopologicalSpace N
    inst✝ : ChartedSpace G N
    n : ENat
    f : M → N
    hf : IsLocalDiffeomorphOn I J n f Set.univ
    ⊢ IsLocalHomeomorphOn f Set.univ
  -/
  exact hf.isLocalHomeomorphOn
  /-
    🎉 no goals
  -/


/-- A local diffeomorphism is an open map. -/
lemma IsLocalDiffeomorph.isOpenMap (hf : IsLocalDiffeomorph I J n f) : IsOpenMap f :=
  (hf.isLocalHomeomorph).isOpenMap


/-- A local diffeomorphism has open range. -/
lemma IsLocalDiffeomorph.isOpen_range (hf : IsLocalDiffeomorph I J n f) : IsOpen (range f) :=
  (hf.isOpenMap).isOpen_range


/-- The image of a local diffeomorphism is open. -/
def IsLocalDiffeomorph.image (hf : IsLocalDiffeomorph I J n f) : Opens N :=
  ⟨range f, hf.isOpen_range⟩


lemma IsLocalDiffeomorph.image_coe (hf : IsLocalDiffeomorph I J n f) : hf.image.1 = range f :=
  rfl

-- TODO: this result holds more generally for (local) structomorphisms
-- This argument implies a `LocalDiffeomorphOn f s` for `s` open is a `PartialDiffeomorph`


/-- A bijective local diffeomorphism is a diffeomorphism. -/
noncomputable def IslocalDiffeomorph.diffeomorph_of_bijective
    (hf : IsLocalDiffeomorph I J n f) (hf' : Function.Bijective f) : Diffeomorph I J M N n := by
  -- Choose a right inverse `g` of `f`.
  /-
    𝕜 : Type u_1
    inst✝¹⁰ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁹ : NormedAddCommGroup E
    inst✝⁸ : NormedSpace 𝕜 E
    F : Type u_3
    inst✝⁷ : NormedAddCommGroup F
    inst✝⁶ : NormedSpace 𝕜 F
    H : Type u_4
    inst✝⁵ : TopologicalSpace H
    G : Type u_5
    inst✝⁴ : TopologicalSpace G
    I : ModelWithCorners 𝕜 E H
    J : ModelWithCorners 𝕜 F G
    M : Type u_6
    inst✝³ : TopologicalSpace M
    inst✝² : ChartedSpace H M
    N : Type u_7
    inst✝¹ : TopologicalSpace N
    inst✝ : ChartedSpace G N
    n : ENat
    f : M → N
    s : Set M
    x : M
    hf : IsLocalDiffeomorph I J n f
    hf' : Function.Bijective f
    ⊢ Diffeomorph I J M N n
  -/
  choose g hgInverse using (Function.bijective_iff_has_inverse).mp hf'
   -- Choose diffeomorphisms φ_x which coincide which `f` near `x`.
  /-
    𝕜 : Type u_1
    inst✝¹⁰ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁹ : NormedAddCommGroup E
    inst✝⁸ : NormedSpace 𝕜 E
    F : Type u_3
    inst✝⁷ : NormedAddCommGroup F
    inst✝⁶ : NormedSpace 𝕜 F
    H : Type u_4
    inst✝⁵ : TopologicalSpace H
    G : Type u_5
    inst✝⁴ : TopologicalSpace G
    I : ModelWithCorners 𝕜 E H
    J : ModelWithCorners 𝕜 F G
    M : Type u_6
    inst✝³ : TopologicalSpace M
    inst✝² : ChartedSpace H M
    N : Type u_7
    inst✝¹ : TopologicalSpace N
    inst✝ : ChartedSpace G N
    n : ENat
    f : M → N
    s : Set M
    x : M
    hf : IsLocalDiffeomorph I J n f
    hf' : Function.Bijective f
    g : N → M
    hgInverse : And (Function.LeftInverse g f) (Function.RightInverse g f)
    ⊢ Diffeomorph I J M N n
  -/
  choose Φ hyp using (fun x ↦ hf x)
  -- Two such diffeomorphisms (and their inverses!) coincide on their sources:
  -- they're both inverses to g. In fact, the latter suffices for our proof.
  -- have (x y) : EqOn (Φ x).symm (Φ y).symm ((Φ x).target ∩ (Φ y).target) := sorry
  have aux (x) : EqOn g (Φ x).symm (Φ x).target :=
    eqOn_of_leftInvOn_of_rightInvOn (fun x' _ ↦ hgInverse.1 x')
      (LeftInvOn.congr_left ((Φ x).toPartialHomeomorph).rightInvOn
        ((Φ x).toPartialHomeomorph).symm_mapsTo (hyp x).2.symm)
      (fun _y hy ↦ (Φ x).map_target hy)
  exact {
    toFun := f
    invFun := g
    left_inv := hgInverse.1
    right_inv := hgInverse.2
    contMDiff_toFun := hf.contMDiff
    contMDiff_invFun := by
      intro y
      let x := g y
      obtain ⟨hx, hfx⟩ := hyp x
      apply ((Φ x).symm.contMDiffOn.congr (aux x)).contMDiffAt (((Φ x).open_target).mem_nhds ?_)
      have : y = (Φ x) x := ((hgInverse.2 y).congr (hfx hx)).mp rfl
      exact this ▸ (Φ x).map_source hx }


