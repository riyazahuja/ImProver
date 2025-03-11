theorem map_continuousAt (f : F) (a : α) : ContinuousAt f a :=
  (map_continuous f).continuousAt


theorem map_continuousWithinAt (f : F) (s : Set α) (a : α) : ContinuousWithinAt f s a :=
  (map_continuous f).continuousWithinAt


/-- Deprecated. Use `map_continuousAt` instead. -/
protected theorem continuousAt (f : C(α, β)) (x : α) : ContinuousAt f x :=
  map_continuousAt f x


theorem map_specializes (f : C(α, β)) {x y : α} (h : x ⤳ y) : f x ⤳ f y :=
  h.map f.2


/--
The continuous functions from `α` to `β` are the same as the plain functions when `α` is discrete.
-/
@[simps]
def equivFnOfDiscrete [DiscreteTopology α] : C(α, β) ≃ (α → β) :=
  ⟨fun f => f,
    fun f => ⟨f, continuous_of_discreteTopology⟩,
                /-
                  α : Type u_1
                  β : Type u_2
                  γ : Type u_3
                  δ : Type u_4
                  inst✝⁴ : TopologicalSpace α
                  inst✝³ : TopologicalSpace β
                  inst✝² : TopologicalSpace γ
                  inst✝¹ : TopologicalSpace δ
                  f g : ContinuousMap α β
                  inst✝ : DiscreteTopology α
                  x✝ : ContinuousMap α β
                  ⊢ Eq ((fun f => { toFun := f, continuous_toFun := ⋯ }) ((fun f => ⇑f) x✝)) x✝
                -/
    fun _ => by ext; rfl,
                     /-
                       🎉 no goals
                     -/
                /-
                  α : Type u_1
                  β : Type u_2
                  γ : Type u_3
                  δ : Type u_4
                  inst✝⁴ : TopologicalSpace α
                  inst✝³ : TopologicalSpace β
                  inst✝² : TopologicalSpace γ
                  inst✝¹ : TopologicalSpace δ
                  f g : ContinuousMap α β
                  inst✝ : DiscreteTopology α
                  x✝ : α → β
                  ⊢ Eq ((fun f => ⇑f) ((fun f => { toFun := f, continuous_toFun := ⋯ }) x✝)) x✝
                -/
    fun _ => by ext; rfl⟩
                     /-
                       🎉 no goals
                     -/


/-- The identity as a continuous map. -/
protected def id : C(α, α) where
  toFun := id


@[simp]
theorem coe_id : ⇑(ContinuousMap.id α) = id :=
  rfl


/-- The constant map as a continuous map. -/
def const (b : β) : C(α, β) where
  toFun := fun _ : α => b


@[simp]
theorem coe_const (b : β) : ⇑(const α b) = Function.const α b :=
  rfl


/-- `Function.const α b` as a bundled continuous function of `b`. -/
@[simps (config := .asFn)]
def constPi : C(β, α → β) where
  toFun b := Function.const α b


instance [Inhabited β] : Inhabited C(α, β) :=
  ⟨const α default⟩


@[simp]
theorem id_apply (a : α) : ContinuousMap.id α a = a :=
  rfl


@[simp]
theorem const_apply (b : β) (a : α) : const α b a = b :=
  rfl


/-- The composition of continuous maps, as a continuous map. -/
def comp (f : C(β, γ)) (g : C(α, β)) : C(α, γ) where
  toFun := f ∘ g


@[simp]
theorem coe_comp (f : C(β, γ)) (g : C(α, β)) : ⇑(comp f g) = f ∘ g :=
  rfl


@[simp]
theorem comp_apply (f : C(β, γ)) (g : C(α, β)) (a : α) : comp f g a = f (g a) :=
  rfl


@[simp]
theorem comp_assoc (f : C(γ, δ)) (g : C(β, γ)) (h : C(α, β)) :
    (f.comp g).comp h = f.comp (g.comp h) :=
  rfl


@[simp]
theorem id_comp (f : C(α, β)) : (ContinuousMap.id _).comp f = f :=
  ext fun _ => rfl


@[simp]
theorem comp_id (f : C(α, β)) : f.comp (ContinuousMap.id _) = f :=
  ext fun _ => rfl


@[simp]
theorem const_comp (c : γ) (f : C(α, β)) : (const β c).comp f = const α c :=
  ext fun _ => rfl


@[simp]
theorem comp_const (f : C(β, γ)) (b : β) : f.comp (const α b) = const α (f b) :=
  ext fun _ => rfl


@[simp]
theorem cancel_right {f₁ f₂ : C(β, γ)} {g : C(α, β)} (hg : Surjective g) :
    f₁.comp g = f₂.comp g ↔ f₁ = f₂ :=
  ⟨fun h => ext <| hg.forall.2 <| DFunLike.ext_iff.1 h, congr_arg (ContinuousMap.comp · g)⟩


@[simp]
theorem cancel_left {f : C(β, γ)} {g₁ g₂ : C(α, β)} (hf : Injective f) :
    f.comp g₁ = f.comp g₂ ↔ g₁ = g₂ :=
                                  /-
                                    α : Type u_1
                                    β : Type u_2
                                    γ : Type u_3
                                    inst✝² : TopologicalSpace α
                                    inst✝¹ : TopologicalSpace β
                                    inst✝ : TopologicalSpace γ
                                    f : ContinuousMap β γ
                                    g₁ g₂ : ContinuousMap α β
                                    hf : Function.Injective ⇑f
                                    h : Eq (f.comp g₁) (f.comp g₂)
                                    a : α
                                    ⊢ Eq (f (g₁ a)) (f (g₂ a))
                                  -/
  ⟨fun h => ext fun a => hf <| by rw [← comp_apply, h, comp_apply], congr_arg _⟩
                                  /-
                                    🎉 no goals
                                  -/


instance [Nonempty α] [Nontrivial β] : Nontrivial C(α, β) :=
  ⟨let ⟨b₁, b₂, hb⟩ := exists_pair_ne β
  ⟨const _ b₁, const _ b₂, fun h => hb <| DFunLike.congr_fun h <| Classical.arbitrary α⟩⟩


/-- `Prod.fst : (x, y) ↦ x` as a bundled continuous map. -/
@[simps (config := .asFn)]
def fst : C(α × β, α) where
  toFun := Prod.fst


/-- `Prod.snd : (x, y) ↦ y` as a bundled continuous map. -/
@[simps (config := .asFn)]
def snd : C(α × β, β) where
  toFun := Prod.snd


/-- Given two continuous maps `f` and `g`, this is the continuous map `x ↦ (f x, g x)`. -/
def prodMk (f : C(α, β₁)) (g : C(α, β₂)) : C(α, β₁ × β₂) where
  toFun x := (f x, g x)


/-- Given two continuous maps `f` and `g`, this is the continuous map `(x, y) ↦ (f x, g y)`. -/
@[simps]
def prodMap (f : C(α₁, α₂)) (g : C(β₁, β₂)) : C(α₁ × β₁, α₂ × β₂) where
  toFun := Prod.map f g


@[simp]
theorem prod_eval (f : C(α, β₁)) (g : C(α, β₂)) (a : α) : (prodMk f g) a = (f a, g a) :=
  rfl


/-- `Prod.swap` bundled as a `ContinuousMap`. -/
@[simps!]
def prodSwap : C(α × β, β × α) := .prodMk .snd .fst


/-- `Sigma.mk i` as a bundled continuous map. -/
@[simps apply]
def sigmaMk (i : I) : C(X i, Σ i, X i) where
  toFun := Sigma.mk i


/--
To give a continuous map out of a disjoint union, it suffices to give a continuous map out of
each term. This is `Sigma.uncurry` for continuous maps.
-/
@[simps]
def sigma (f : ∀ i, C(X i, A)) : C((Σ i, X i), A) where
  toFun ig := f ig.fst ig.snd


variable (A X) in
/--
Giving a continuous map out of a disjoint union is the same as giving a continuous map out of
each term. This is a version of `Equiv.piCurry` for continuous maps.
-/
@[simps]
def sigmaEquiv : (∀ i, C(X i, A)) ≃ C((Σ i, X i), A) where
  toFun := sigma
  invFun f i := f.comp (sigmaMk i)
                 /-
                   α : Type u_1
                   β : Type u_2
                   γ : Type u_3
                   δ : Type u_4
                   inst✝⁵ : TopologicalSpace α
                   inst✝⁴ : TopologicalSpace β
                   inst✝³ : TopologicalSpace γ
                   inst✝² : TopologicalSpace δ
                   f g : ContinuousMap α β
                   I : Type u_5
                   A : Type u_6
                   X : I → Type u_7
                   inst✝¹ : TopologicalSpace A
                   inst✝ : (i : I) → TopologicalSpace (X i)
                   ⊢ Function.LeftInverse (fun f i => f.comp (ContinuousMap.sigmaMk i)) Continuou …
                 -/
  left_inv := by intro; ext; simp
                             /-
                               🎉 no goals
                             -/
                  /-
                    α : Type u_1
                    β : Type u_2
                    γ : Type u_3
                    δ : Type u_4
                    inst✝⁵ : TopologicalSpace α
                    inst✝⁴ : TopologicalSpace β
                    inst✝³ : TopologicalSpace γ
                    inst✝² : TopologicalSpace δ
                    f g : ContinuousMap α β
                    I : Type u_5
                    A : Type u_6
                    X : I → Type u_7
                    inst✝¹ : TopologicalSpace A
                    inst✝ : (i : I) → TopologicalSpace (X i)
                    ⊢ Function.RightInverse (fun f i => f.comp (ContinuousMap.sigmaMk i)) Continuo …
                  -/
  right_inv := by intro; ext; simp
                              /-
                                🎉 no goals
                              -/


/-- Abbreviation for product of continuous maps, which is continuous -/
def pi (f : ∀ i, C(A, X i)) : C(A, ∀ i, X i) where
  toFun (a : A) (i : I) := f i a


@[simp]
theorem pi_eval (f : ∀ i, C(A, X i)) (a : A) : (pi f) a = fun i : I => (f i) a :=
  rfl


/-- Evaluation at point as a bundled continuous map. -/
@[simps (config := .asFn)]
def eval (i : I) : C(∀ j, X j, X i) where
  toFun := Function.eval i


variable (A X) in
/--
Giving a continuous map out of a disjoint union is the same as giving a continuous map out of
each term
-/
@[simps]
def piEquiv : (∀ i, C(A, X i)) ≃ C(A, ∀ i, X i) where
  toFun := pi
  invFun f i := (eval i).comp f
                 /-
                   α : Type u_1
                   β : Type u_2
                   γ : Type u_3
                   δ : Type u_4
                   inst✝⁶ : TopologicalSpace α
                   inst✝⁵ : TopologicalSpace β
                   inst✝⁴ : TopologicalSpace γ
                   inst✝³ : TopologicalSpace δ
                   f g : ContinuousMap α β
                   I : Type u_5
                   A : Type u_6
                   X : I → Type u_7
                   Y : I → Type u_8
                   inst✝² : TopologicalSpace A
                   inst✝¹ : (i : I) → TopologicalSpace (X i)
                   inst✝ : (i : I) → TopologicalSpace (Y i)
                   ⊢ Function.LeftInverse (fun f i => (ContinuousMap.eval i).comp f) ContinuousMa …
                 -/
  left_inv := by intro; ext; simp [pi]
                             /-
                               🎉 no goals
                             -/
                  /-
                    α : Type u_1
                    β : Type u_2
                    γ : Type u_3
                    δ : Type u_4
                    inst✝⁶ : TopologicalSpace α
                    inst✝⁵ : TopologicalSpace β
                    inst✝⁴ : TopologicalSpace γ
                    inst✝³ : TopologicalSpace δ
                    f g : ContinuousMap α β
                    I : Type u_5
                    A : Type u_6
                    X : I → Type u_7
                    Y : I → Type u_8
                    inst✝² : TopologicalSpace A
                    inst✝¹ : (i : I) → TopologicalSpace (X i)
                    inst✝ : (i : I) → TopologicalSpace (Y i)
                    ⊢ Function.RightInverse (fun f i => (ContinuousMap.eval i).comp f) ContinuousM …
                  -/
  right_inv := by intro; ext; simp [pi]
                              /-
                                🎉 no goals
                              -/


/-- Combine a collection of bundled continuous maps `C(X i, Y i)` into a bundled continuous map
`C(∀ i, X i, ∀ i, Y i)`. -/
@[simps!]
def piMap (f : ∀ i, C(X i, Y i)) : C((i : I) → X i, (i : I) → Y i) :=
  .pi fun i ↦ (f i).comp (eval i)


/-- "Precomposition" as a continuous map between dependent types. -/
def precomp {ι : Type*} (φ : ι → I) : C((i : I) → X i, (i : ι) → X (φ i)) :=
  ⟨_, Pi.continuous_precomp' φ⟩


/-- The restriction of a continuous function `α → β` to a subset `s` of `α`. -/
def restrict (f : C(α, β)) : C(s, β) where
  toFun := f ∘ ((↑) : s → α)


@[simp]
theorem coe_restrict (f : C(α, β)) : ⇑(f.restrict s) = f ∘ ((↑) : s → α) :=
  rfl


@[simp]
theorem restrict_apply (f : C(α, β)) (s : Set α) (x : s) : f.restrict s x = f x :=
  rfl


@[simp]
theorem restrict_apply_mk (f : C(α, β)) (s : Set α) (x : α) (hx : x ∈ s) :
    f.restrict s ⟨x, hx⟩ = f x :=
  rfl


theorem injective_restrict [T2Space β] {s : Set α} (hs : Dense s) :
    Injective (restrict s : C(α, β) → C(s, β)) := fun f g h ↦
  DFunLike.ext' <| (map_continuous f).ext_on hs (map_continuous g) <|
    Set.restrict_eq_restrict_iff.1 <| congr_arg DFunLike.coe h


/-- The restriction of a continuous map to the preimage of a set. -/
@[simps]
def restrictPreimage (f : C(α, β)) (s : Set β) : C(f ⁻¹' s, s) :=
  ⟨s.restrictPreimage f, continuous_iff_continuousAt.mpr fun _ ↦
    (map_continuousAt f _).restrictPreimage⟩


/-- A family `φ i` of continuous maps `C(S i, β)`, where the domains `S i` contain a neighbourhood
of each point in `α` and the functions `φ i` agree pairwise on intersections, can be glued to
construct a continuous map in `C(α, β)`. -/
noncomputable def liftCover : C(α, β) :=
  haveI H : ⋃ i, S i = Set.univ :=
    Set.iUnion_eq_univ_iff.2 fun x ↦ (hS x).imp fun _ ↦ mem_of_mem_nhds
  mk (Set.liftCover S (fun i ↦ φ i) hφ H) <| continuous_of_cover_nhds hS fun i ↦ by
    /-
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      δ : Type u_4
      inst✝³ : TopologicalSpace α
      inst✝² : TopologicalSpace β
      inst✝¹ : TopologicalSpace γ
      inst✝ : TopologicalSpace δ
      f g : ContinuousMap α β
      ι : Type u_5
      S : ι → Set α
      φ : (i : ι) → ContinuousMap (↑(S i)) β
      hφ : ∀ (i j : ι) (x : α) (hxi : Membership.mem (S i) x) (hxj : Membership.mem  …
      hS : ∀ (x : α), Exists fun i => Membership.mem (nhds x) (S i)
      H : Eq (Set.iUnion fun i => S i) Set.univ
      i : ι
      ⊢ ContinuousOn (Set.liftCover S (fun i => ⇑(φ i)) hφ H) (S i)
    -/
    rw [continuousOn_iff_continuous_restrict]
    simpa (config := { unfoldPartialApp := true }) only [Set.restrict, Set.liftCover_coe]
      using map_continuous (φ i)


@[simp]
theorem liftCover_coe {i : ι} (x : S i) : liftCover S φ hφ hS x = φ i x := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSpace β
    ι : Type u_5
    S : ι → Set α
    φ : (i : ι) → ContinuousMap (↑(S i)) β
    hφ : ∀ (i j : ι) (x : α) (hxi : Membership.mem (S i) x) (hxj : Membership.mem  …
    hS : ∀ (x : α), Exists fun i => Membership.mem (nhds x) (S i)
    i : ι
    x : ↑(S i)
    ⊢ Eq ((ContinuousMap.liftCover S φ hφ hS) ↑x) ((φ i) x)
  -/
  rw [liftCover, coe_mk, Set.liftCover_coe _]
  /-
    🎉 no goals
  -/


@[simp]
theorem liftCover_restrict {i : ι} : (liftCover S φ hφ hS).restrict (S i) = φ i := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSpace β
    ι : Type u_5
    S : ι → Set α
    φ : (i : ι) → ContinuousMap (↑(S i)) β
    hφ : ∀ (i j : ι) (x : α) (hxi : Membership.mem (S i) x) (hxj : Membership.mem  …
    hS : ∀ (x : α), Exists fun i => Membership.mem (nhds x) (S i)
    i : ι
    ⊢ Eq (ContinuousMap.restrict (S i) (ContinuousMap.liftCover S φ hφ hS)) (φ i)
  -/
  ext
  /-
    case h
    α : Type u_1
    β : Type u_2
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSpace β
    ι : Type u_5
    S : ι → Set α
    φ : (i : ι) → ContinuousMap (↑(S i)) β
    hφ : ∀ (i j : ι) (x : α) (hxi : Membership.mem (S i) x) (hxj : Membership.mem  …
    hS : ∀ (x : α), Exists fun i => Membership.mem (nhds x) (S i)
    i : ι
    a✝ : ↑(S i)
    ⊢ Eq ((ContinuousMap.restrict (S i) (ContinuousMap.liftCover S φ hφ hS)) a✝) ( …
  -/
  simp only [coe_restrict, Function.comp_apply, liftCover_coe]
  /-
    🎉 no goals
  -/


/-- A family `F s` of continuous maps `C(s, β)`, where (1) the domains `s` are taken from a set `A`
of sets in `α` which contain a neighbourhood of each point in `α` and (2) the functions `F s` agree
pairwise on intersections, can be glued to construct a continuous map in `C(α, β)`. -/
noncomputable def liftCover' : C(α, β) := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    δ : Type u_4
    inst✝³ : TopologicalSpace α
    inst✝² : TopologicalSpace β
    inst✝¹ : TopologicalSpace γ
    inst✝ : TopologicalSpace δ
    f g : ContinuousMap α β
    ι : Type u_5
    S : ι → Set α
    φ : (i : ι) → ContinuousMap (↑(S i)) β
    hφ : ∀ (i j : ι) (x : α) (hxi : Membership.mem (S i) x) (hxj : Membership.mem  …
    hS : ∀ (x : α), Exists fun i => Membership.mem (nhds x) (S i)
    A : Set (Set α)
    F : (s : Set α) → Membership.mem A s → ContinuousMap (↑s) β
    hF : ∀ (s : Set α) (hs : Membership.mem A s) (t : Set α) (ht : Membership.mem  …
    hA : ∀ (x : α), Exists fun i => And (Membership.mem A i) (Membership.mem (nhds …
    ⊢ ContinuousMap α β
  -/
  let S : A → Set α := (↑)
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    δ : Type u_4
    inst✝³ : TopologicalSpace α
    inst✝² : TopologicalSpace β
    inst✝¹ : TopologicalSpace γ
    inst✝ : TopologicalSpace δ
    f g : ContinuousMap α β
    ι : Type u_5
    S✝ : ι → Set α
    φ : (i : ι) → ContinuousMap (↑(S✝ i)) β
    hφ : ∀ (i j : ι) (x : α) (hxi : Membership.mem (S✝ i) x) (hxj : Membership.mem …
    hS : ∀ (x : α), Exists fun i => Membership.mem (nhds x) (S✝ i)
    A : Set (Set α)
    F : (s : Set α) → Membership.mem A s → ContinuousMap (↑s) β
    hF : ∀ (s : Set α) (hs : Membership.mem A s) (t : Set α) (ht : Membership.mem  …
    hA : ∀ (x : α), Exists fun i => And (Membership.mem A i) (Membership.mem (nhds …
    S : ↑A → Set α := Subtype.val
    ⊢ ContinuousMap α β
  -/
  let F : ∀ i : A, C(i, β) := fun i => F i i.prop
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    δ : Type u_4
    inst✝³ : TopologicalSpace α
    inst✝² : TopologicalSpace β
    inst✝¹ : TopologicalSpace γ
    inst✝ : TopologicalSpace δ
    f g : ContinuousMap α β
    ι : Type u_5
    S✝ : ι → Set α
    φ : (i : ι) → ContinuousMap (↑(S✝ i)) β
    hφ : ∀ (i j : ι) (x : α) (hxi : Membership.mem (S✝ i) x) (hxj : Membership.mem …
    hS : ∀ (x : α), Exists fun i => Membership.mem (nhds x) (S✝ i)
    A : Set (Set α)
    F✝ : (s : Set α) → Membership.mem A s → ContinuousMap (↑s) β
    hF : ∀ (s : Set α) (hs : Membership.mem A s) (t : Set α) (ht : Membership.mem  …
    hA : ∀ (x : α), Exists fun i => And (Membership.mem A i) (Membership.mem (nhds …
    S : ↑A → Set α := Subtype.val
    F : (i : ↑A) → ContinuousMap (↑↑i) β := fun i => F✝ ↑i ⋯
    ⊢ ContinuousMap α β
  -/
  refine liftCover S F (fun i j => hF i i.prop j j.prop) ?_
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    δ : Type u_4
    inst✝³ : TopologicalSpace α
    inst✝² : TopologicalSpace β
    inst✝¹ : TopologicalSpace γ
    inst✝ : TopologicalSpace δ
    f g : ContinuousMap α β
    ι : Type u_5
    S✝ : ι → Set α
    φ : (i : ι) → ContinuousMap (↑(S✝ i)) β
    hφ : ∀ (i j : ι) (x : α) (hxi : Membership.mem (S✝ i) x) (hxj : Membership.mem …
    hS : ∀ (x : α), Exists fun i => Membership.mem (nhds x) (S✝ i)
    A : Set (Set α)
    F✝ : (s : Set α) → Membership.mem A s → ContinuousMap (↑s) β
    hF : ∀ (s : Set α) (hs : Membership.mem A s) (t : Set α) (ht : Membership.mem  …
    hA : ∀ (x : α), Exists fun i => And (Membership.mem A i) (Membership.mem (nhds …
    S : ↑A → Set α := Subtype.val
    F : (i : ↑A) → ContinuousMap (↑↑i) β := fun i => F✝ ↑i ⋯
    ⊢ ∀ (x : α), Exists fun i => Membership.mem (nhds x) (S i)
  -/
  intro x
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    δ : Type u_4
    inst✝³ : TopologicalSpace α
    inst✝² : TopologicalSpace β
    inst✝¹ : TopologicalSpace γ
    inst✝ : TopologicalSpace δ
    f g : ContinuousMap α β
    ι : Type u_5
    S✝ : ι → Set α
    φ : (i : ι) → ContinuousMap (↑(S✝ i)) β
    hφ : ∀ (i j : ι) (x : α) (hxi : Membership.mem (S✝ i) x) (hxj : Membership.mem …
    hS : ∀ (x : α), Exists fun i => Membership.mem (nhds x) (S✝ i)
    A : Set (Set α)
    F✝ : (s : Set α) → Membership.mem A s → ContinuousMap (↑s) β
    hF : ∀ (s : Set α) (hs : Membership.mem A s) (t : Set α) (ht : Membership.mem  …
    hA : ∀ (x : α), Exists fun i => And (Membership.mem A i) (Membership.mem (nhds …
    S : ↑A → Set α := Subtype.val
    F : (i : ↑A) → ContinuousMap (↑↑i) β := fun i => F✝ ↑i ⋯
    x : α
    ⊢ Exists fun i => Membership.mem (nhds x) (S i)
  -/
  obtain ⟨s, hs, hsx⟩ := hA x
  /-
    case intro.intro
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    δ : Type u_4
    inst✝³ : TopologicalSpace α
    inst✝² : TopologicalSpace β
    inst✝¹ : TopologicalSpace γ
    inst✝ : TopologicalSpace δ
    f g : ContinuousMap α β
    ι : Type u_5
    S✝ : ι → Set α
    φ : (i : ι) → ContinuousMap (↑(S✝ i)) β
    hφ : ∀ (i j : ι) (x : α) (hxi : Membership.mem (S✝ i) x) (hxj : Membership.mem …
    hS : ∀ (x : α), Exists fun i => Membership.mem (nhds x) (S✝ i)
    A : Set (Set α)
    F✝ : (s : Set α) → Membership.mem A s → ContinuousMap (↑s) β
    hF : ∀ (s : Set α) (hs : Membership.mem A s) (t : Set α) (ht : Membership.mem  …
    hA : ∀ (x : α), Exists fun i => And (Membership.mem A i) (Membership.mem (nhds …
    S : ↑A → Set α := Subtype.val
    F : (i : ↑A) → ContinuousMap (↑↑i) β := fun i => F✝ ↑i ⋯
    x : α
    s : Set α
    hs : Membership.mem A s
    hsx : Membership.mem (nhds x) s
    ⊢ Exists fun i => Membership.mem (nhds x) (S i)
  -/
  exact ⟨⟨s, hs⟩, hsx⟩
  /-
    🎉 no goals
  -/


@[simp]
theorem liftCover_coe' {s : Set α} {hs : s ∈ A} (x : s) : liftCover' A F hF hA x = F s hs x :=
  let x' : ((↑) : A → Set α) ⟨s, hs⟩ := x
     /-
       α : Type u_1
       β : Type u_2
       inst✝¹ : TopologicalSpace α
       inst✝ : TopologicalSpace β
       A : Set (Set α)
       F : (s : Set α) → Membership.mem A s → ContinuousMap (↑s) β
       hF : ∀ (s : Set α) (hs : Membership.mem A s) (t : Set α) (ht : Membership.mem  …
       hA : ∀ (x : α), Exists fun i => And (Membership.mem A i) (Membership.mem (nhds …
       s : Set α
       hs : Membership.mem A s
       x : ↑s
       x' : ↑↑⟨s, hs⟩ := x
       ⊢ Eq ((ContinuousMap.liftCover' A F hF hA) ↑x) ((F s hs) x)
     -/
  by delta liftCover'; exact liftCover_coe x'
                       /-
                         🎉 no goals
                       -/

-- Porting note: porting program suggested `ext <| liftCover_coe'`

@[simp]
theorem liftCover_restrict' {s : Set α} {hs : s ∈ A} :
    (liftCover' A F hF hA).restrict s = F s hs := ext <| liftCover_coe' (hF := hF) (hA := hA)


/-- `Set.inclusion` as a bundled continuous map. -/
def inclusion {s t : Set α} (h : s ⊆ t) : C(s, t) where
  toFun := Set.inclusion h
  continuous_toFun := continuous_inclusion h


/-- `Setoid.quotientKerEquivOfRightInverse` as a homeomorphism. -/
@[simps!]
def Function.RightInverse.homeomorph {f' : C(Y, X)} (hf : Function.RightInverse f' f) :
    Quotient (Setoid.ker f) ≃ₜ Y where
  toEquiv := Setoid.quotientKerEquivOfRightInverse _ _ hf
  continuous_toFun := isQuotientMap_quot_mk.continuous_iff.mpr (map_continuous f)
  continuous_invFun := continuous_quotient_mk'.comp (map_continuous f')


/--
The homeomorphism from the quotient of a quotient map to its codomain. This is
`Setoid.quotientKerEquivOfSurjective` as a homeomorphism.
-/
@[simps!]
noncomputable def homeomorph (hf : IsQuotientMap f) : Quotient (Setoid.ker f) ≃ₜ Y where
  toEquiv := Setoid.quotientKerEquivOfSurjective _ hf.surjective
  continuous_toFun := isQuotientMap_quot_mk.continuous_iff.mpr hf.continuous
  continuous_invFun := by
    /-
      X : Type u_1
      Y : Type u_2
      Z : Type u_3
      inst✝² : TopologicalSpace X
      inst✝¹ : TopologicalSpace Y
      inst✝ : TopologicalSpace Z
      f : ContinuousMap X Y
      hf : Topology.IsQuotientMap ⇑f
      ⊢ Continuous (Setoid.quotientKerEquivOfSurjective ⇑f ⋯).invFun
    -/
    rw [hf.continuous_iff]
    /-
      X : Type u_1
      Y : Type u_2
      Z : Type u_3
      inst✝² : TopologicalSpace X
      inst✝¹ : TopologicalSpace Y
      inst✝ : TopologicalSpace Z
      f : ContinuousMap X Y
      hf : Topology.IsQuotientMap ⇑f
      ⊢ Continuous (Function.comp (Setoid.quotientKerEquivOfSurjective ⇑f ⋯).invFun  …
    -/
    convert continuous_quotient_mk'
    /-
      case h.e'_5
      X : Type u_1
      Y : Type u_2
      Z : Type u_3
      inst✝² : TopologicalSpace X
      inst✝¹ : TopologicalSpace Y
      inst✝ : TopologicalSpace Z
      f : ContinuousMap X Y
      hf : Topology.IsQuotientMap ⇑f
      ⊢ Eq (Function.comp (Setoid.quotientKerEquivOfSurjective ⇑f ⋯).invFun ⇑f) Quot …
    -/
    ext
    simp only [Equiv.invFun_as_coe, Function.comp_apply,
      (Setoid.quotientKerEquivOfSurjective f hf.surjective).symm_apply_eq]
    /-
      case h.e'_5.h
      X : Type u_1
      Y : Type u_2
      Z : Type u_3
      inst✝² : TopologicalSpace X
      inst✝¹ : TopologicalSpace Y
      inst✝ : TopologicalSpace Z
      f : ContinuousMap X Y
      hf : Topology.IsQuotientMap ⇑f
      x✝ : X
      ⊢ Eq (f x✝) ((Setoid.quotientKerEquivOfSurjective ⇑f ⋯) (Quotient.mk' x✝))
    -/
    rfl
    /-
      🎉 no goals
    -/


/-- Descend a continuous map, which is constant on the fibres, along a quotient map. -/
@[simps]
noncomputable def lift : C(Y, Z) where
  toFun := ((fun i ↦ Quotient.liftOn' i g (fun _ _ (hab : f _ = f _) ↦ h hab)) :
    Quotient (Setoid.ker f) → Z) ∘ hf.homeomorph.symm
  continuous_toFun := Continuous.comp (continuous_quot_lift _ g.2) (Homeomorph.continuous _)


/--
The obvious triangle induced by `IsQuotientMap.lift` commutes:
```
     g
  X --→ Z
  |   ↗
f |  / hf.lift g h
  v /
  Y
```
-/
@[simp]
theorem lift_comp : (hf.lift g h).comp f = g := by
  /-
    X : Type u_1
    Y : Type u_2
    Z : Type u_3
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    inst✝ : TopologicalSpace Z
    f : ContinuousMap X Y
    hf : Topology.IsQuotientMap ⇑f
    g : ContinuousMap X Z
    h : Function.FactorsThrough ⇑g ⇑f
    ⊢ Eq ((hf.lift g h).comp f) g
  -/
  ext
  /-
    case h
    X : Type u_1
    Y : Type u_2
    Z : Type u_3
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    inst✝ : TopologicalSpace Z
    f : ContinuousMap X Y
    hf : Topology.IsQuotientMap ⇑f
    g : ContinuousMap X Z
    h : Function.FactorsThrough ⇑g ⇑f
    a✝ : X
    ⊢ Eq (((hf.lift g h).comp f) a✝) (g a✝)
  -/
  simpa using h (Function.rightInverse_surjInv _ _)
  /-
    🎉 no goals
  -/


/-- `IsQuotientMap.lift` as an equivalence. -/
@[simps]
noncomputable def liftEquiv : { g : C(X, Z) // Function.FactorsThrough g f} ≃ C(Y, Z) where
  toFun g := hf.lift g g.prop
                                        /-
                                          X : Type u_1
                                          Y : Type u_2
                                          Z : Type u_3
                                          inst✝² : TopologicalSpace X
                                          inst✝¹ : TopologicalSpace Y
                                          inst✝ : TopologicalSpace Z
                                          f : ContinuousMap X Y
                                          hf : Topology.IsQuotientMap ⇑f
                                          g✝ : ContinuousMap X Z
                                          h✝ : Function.FactorsThrough ⇑g✝ ⇑f
                                          g : ContinuousMap Y Z
                                          x✝¹ x✝ : X
                                          h : Eq (f x✝¹) (f x✝)
                                          ⊢ Eq ((g.comp f) x✝¹) ((g.comp f) x✝)
                                        -/
  invFun g := ⟨g.comp f, fun _ _ h ↦ by simp only [ContinuousMap.comp_apply]; rw [h]⟩
                                                                              /-
                                                                                🎉 no goals
                                                                              -/
                 /-
                   X : Type u_1
                   Y : Type u_2
                   Z : Type u_3
                   inst✝² : TopologicalSpace X
                   inst✝¹ : TopologicalSpace Y
                   inst✝ : TopologicalSpace Z
                   f : ContinuousMap X Y
                   hf : Topology.IsQuotientMap ⇑f
                   g : ContinuousMap X Z
                   h : Function.FactorsThrough ⇑g ⇑f
                   ⊢ Function.LeftInverse (fun g => ⟨g.comp f, ⋯⟩) fun g => hf.lift ↑g ⋯
                 -/
  left_inv := by intro; simp
                        /-
                          🎉 no goals
                        -/
  right_inv := by
    /-
      X : Type u_1
      Y : Type u_2
      Z : Type u_3
      inst✝² : TopologicalSpace X
      inst✝¹ : TopologicalSpace Y
      inst✝ : TopologicalSpace Z
      f : ContinuousMap X Y
      hf : Topology.IsQuotientMap ⇑f
      g : ContinuousMap X Z
      h : Function.FactorsThrough ⇑g ⇑f
      ⊢ Function.RightInverse (fun g => ⟨g.comp f, ⋯⟩) fun g => hf.lift ↑g ⋯
    -/
    intro g
    /-
      X : Type u_1
      Y : Type u_2
      Z : Type u_3
      inst✝² : TopologicalSpace X
      inst✝¹ : TopologicalSpace Y
      inst✝ : TopologicalSpace Z
      f : ContinuousMap X Y
      hf : Topology.IsQuotientMap ⇑f
      g✝ : ContinuousMap X Z
      h : Function.FactorsThrough ⇑g✝ ⇑f
      g : ContinuousMap Y Z
      ⊢ Eq ((fun g => hf.lift ↑g ⋯) ((fun g => ⟨g.comp f, ⋯⟩) g)) g
    -/
    ext a
    /-
      case h
      X : Type u_1
      Y : Type u_2
      Z : Type u_3
      inst✝² : TopologicalSpace X
      inst✝¹ : TopologicalSpace Y
      inst✝ : TopologicalSpace Z
      f : ContinuousMap X Y
      hf : Topology.IsQuotientMap ⇑f
      g✝ : ContinuousMap X Z
      h : Function.FactorsThrough ⇑g✝ ⇑f
      g : ContinuousMap Y Z
      a : Y
      ⊢ Eq (((fun g => hf.lift ↑g ⋯) ((fun g => ⟨g.comp f, ⋯⟩) g)) a) (g a)
    -/
    simpa using congrArg g (Function.rightInverse_surjInv hf.surjective a)
    /-
      🎉 no goals
    -/


instance instContinuousMapClass : ContinuousMapClass (α ≃ₜ β) α β where
  map_continuous f := f.continuous_toFun


/-- The forward direction of a homeomorphism, as a bundled continuous map. -/
@[simps, deprecated _root_.toContinuousMap (since := "2024-10-12")]
protected def toContinuousMap (e : α ≃ₜ β) : C(α, β) :=
  ⟨e, e.continuous_toFun⟩


@[simp]
theorem coe_refl : (Homeomorph.refl α : C(α, α)) = ContinuousMap.id α :=
  rfl


@[simp]
theorem coe_trans : (f.trans g : C(α, γ)) = (g : C(β, γ)).comp f :=
  rfl


/-- Left inverse to a continuous map from a homeomorphism, mirroring `Equiv.symm_comp_self`. -/
@[simp]
theorem symm_comp_toContinuousMap :
    (f.symm : C(β, α)).comp (f : C(α, β)) = ContinuousMap.id α := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSpace β
    f : Homeomorph α β
    ⊢ Eq ((↑f.symm).comp ↑f) (ContinuousMap.id α)
  -/
  rw [← coe_trans, self_trans_symm, coe_refl]
  /-
    🎉 no goals
  -/


/-- Right inverse to a continuous map from a homeomorphism, mirroring `Equiv.self_comp_symm`. -/
@[simp]
theorem toContinuousMap_comp_symm :
    (f : C(α, β)).comp (f.symm : C(β, α)) = ContinuousMap.id β := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSpace β
    f : Homeomorph α β
    ⊢ Eq ((↑f).comp ↑f.symm) (ContinuousMap.id β)
  -/
  rw [← coe_trans, symm_trans_self, coe_refl]
  /-
    🎉 no goals
  -/


