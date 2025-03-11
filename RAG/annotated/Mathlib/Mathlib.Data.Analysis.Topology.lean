/-- A `Ctop α σ` is a realization of a topology (basis) on `α`,
  represented by a type `σ` together with operations for the top element and
  the intersection operation. -/
structure Ctop (α σ : Type*) where
  f : σ → Set α
  top : α → σ
  top_mem : ∀ x : α, x ∈ f (top x)
  inter : ∀ (a b) (x : α), x ∈ f a ∩ f b → σ
  inter_mem : ∀ a b x h, x ∈ f (inter a b x h)
  inter_sub : ∀ a b x h, f (inter a b x h) ⊆ f a ∩ f b


instance : Inhabited (Ctop α (Set α)) :=
  ⟨{  f := id
      top := singleton
      top_mem := mem_singleton
      inter := fun s t _ _ ↦ s ∩ t
      inter_mem := fun _s _t _a ↦ id
      inter_sub := fun _s _t _a _ha ↦ Subset.rfl }⟩


instance : CoeFun (Ctop α σ) fun _ ↦ σ → Set α :=
  ⟨Ctop.f⟩

-- @[simp] -- Porting note (https://github.com/leanprover-community/mathlib4/issues/10685): dsimp can prove this

theorem coe_mk (f T h₁ I h₂ h₃ a) : (@Ctop.mk α σ f T h₁ I h₂ h₃) a = f a := rfl


/-- Map a Ctop to an equivalent representation type. -/
def ofEquiv (E : σ ≃ τ) : Ctop α σ → Ctop α τ
  | ⟨f, T, h₁, I, h₂, h₃⟩ =>
    { f := fun a ↦ f (E.symm a)
      top := fun x ↦ E (T x)
                            /-
                              α : Type u_1
                              β : Type u_2
                              σ : Type u_3
                              τ : Type u_4
                              F : Ctop α σ
                              E : Equiv σ τ
                              f : σ → Set α
                              T : α → σ
                              h₁ : ∀ (x : α), Membership.mem (f (T x)) x
                              I : (a b : σ) → (x : α) → Membership.mem (Inter.inter (f a) (f b)) x → σ
                              h₂ : ∀ (a b : σ) (x : α) (h : Membership.mem (Inter.inter (f a) (f b)) x), Mem …
                              h₃ : ∀ (a b : σ) (x : α) (h : Membership.mem (Inter.inter (f a) (f b)) x), Has …
                              x : α
                              ⊢ Membership.mem ((fun a => f (E.symm a)) ((fun x => E (T x)) x)) x
                            -/
      top_mem := fun x ↦ by simpa using h₁ x
                            /-
                              🎉 no goals
                            -/
      inter := fun a b x h ↦ E (I (E.symm a) (E.symm b) x h)
                                    /-
                                      α : Type u_1
                                      β : Type u_2
                                      σ : Type u_3
                                      τ : Type u_4
                                      F : Ctop α σ
                                      E : Equiv σ τ
                                      f : σ → Set α
                                      T : α → σ
                                      h₁ : ∀ (x : α), Membership.mem (f (T x)) x
                                      I : (a b : σ) → (x : α) → Membership.mem (Inter.inter (f a) (f b)) x → σ
                                      h₂ : ∀ (a b : σ) (x : α) (h : Membership.mem (Inter.inter (f a) (f b)) x), Mem …
                                      h₃ : ∀ (a b : σ) (x : α) (h : Membership.mem (Inter.inter (f a) (f b)) x), Has …
                                      a b : τ
                                      x : α
                                      h : Membership.mem (Inter.inter ((fun a => f (E.symm a)) a) ((fun a => f (E.sy …
                                      ⊢ Membership.mem ((fun a => f (E.symm a)) ((fun a b x h => E (I (E.symm a) (E. …
                                    -/
      inter_mem := fun a b x h ↦ by simpa using h₂ (E.symm a) (E.symm b) x h
                                    /-
                                      🎉 no goals
                                    -/
                                    /-
                                      α : Type u_1
                                      β : Type u_2
                                      σ : Type u_3
                                      τ : Type u_4
                                      F : Ctop α σ
                                      E : Equiv σ τ
                                      f : σ → Set α
                                      T : α → σ
                                      h₁ : ∀ (x : α), Membership.mem (f (T x)) x
                                      I : (a b : σ) → (x : α) → Membership.mem (Inter.inter (f a) (f b)) x → σ
                                      h₂ : ∀ (a b : σ) (x : α) (h : Membership.mem (Inter.inter (f a) (f b)) x), Mem …
                                      h₃ : ∀ (a b : σ) (x : α) (h : Membership.mem (Inter.inter (f a) (f b)) x), Has …
                                      a b : τ
                                      x : α
                                      h : Membership.mem (Inter.inter ((fun a => f (E.symm a)) a) ((fun a => f (E.sy …
                                      ⊢ HasSubset.Subset ((fun a => f (E.symm a)) ((fun a b x h => E (I (E.symm a) ( …
                                    -/
      inter_sub := fun a b x h ↦ by simpa using h₃ (E.symm a) (E.symm b) x h }
                                    /-
                                      🎉 no goals
                                    -/


@[simp]
theorem ofEquiv_val (E : σ ≃ τ) (F : Ctop α σ) (a : τ) : F.ofEquiv E a = F (E.symm a) := by
  /-
    α : Type u_1
    σ : Type u_3
    τ : Type u_4
    E : Equiv σ τ
    F : Ctop α σ
    a : τ
    ⊢ Eq ((Ctop.ofEquiv E F).f a) (F.f (E.symm a))
  -/
  cases F; rfl
           /-
             🎉 no goals
           -/


/-- Every `Ctop` is a topological space. -/
def toTopsp (F : Ctop α σ) : TopologicalSpace α := TopologicalSpace.generateFrom (Set.range F.f)


theorem toTopsp_isTopologicalBasis (F : Ctop α σ) :
    @TopologicalSpace.IsTopologicalBasis _ F.toTopsp (Set.range F.f) :=
  letI := F.toTopsp
  ⟨fun _u ⟨a, e₁⟩ _v ⟨b, e₂⟩ ↦
    e₁ ▸ e₂ ▸ fun x h ↦ ⟨_, ⟨_, rfl⟩, F.inter_mem a b x h, F.inter_sub a b x h⟩,
    eq_univ_iff_forall.2 fun x ↦ ⟨_, ⟨_, rfl⟩, F.top_mem x⟩, rfl⟩


@[simp]
theorem mem_nhds_toTopsp (F : Ctop α σ) {s : Set α} {a : α} :
    s ∈ @nhds _ F.toTopsp a ↔ ∃ b, a ∈ F b ∧ F b ⊆ s :=
  (@TopologicalSpace.IsTopologicalBasis.mem_nhds_iff _ F.toTopsp _ _ _
        F.toTopsp_isTopologicalBasis).trans <|
    ⟨fun ⟨_, ⟨x, rfl⟩, h⟩ ↦ ⟨x, h⟩, fun ⟨x, h⟩ ↦ ⟨_, ⟨x, rfl⟩, h⟩⟩


/-- A `Ctop` realizer for the topological space `T` is a `Ctop`
  which generates `T`. -/
structure Ctop.Realizer (α) [T : TopologicalSpace α] where
  σ : Type*
  F : Ctop α σ
  eq : F.toTopsp = T


/-- A `Ctop` realizes the topological space it generates. -/
protected def Ctop.toRealizer (F : Ctop α σ) : @Ctop.Realizer _ F.toTopsp :=
  @Ctop.Realizer.mk _ F.toTopsp σ F rfl


instance (F : Ctop α σ) : Inhabited (@Ctop.Realizer _ F.toTopsp) :=
  ⟨F.toRealizer⟩


protected theorem is_basis [T : TopologicalSpace α] (F : Realizer α) :
    TopologicalSpace.IsTopologicalBasis (Set.range F.F.f) := by
  /-
    α : Type u_1
    T : TopologicalSpace α
    F : Ctop.Realizer α
    ⊢ TopologicalSpace.IsTopologicalBasis (Set.range F.F.f)
  -/
  have := toTopsp_isTopologicalBasis F.F; rwa [F.eq] at this
                                          /-
                                            🎉 no goals
                                          -/


protected theorem mem_nhds [T : TopologicalSpace α] (F : Realizer α) {s : Set α} {a : α} :
    s ∈ 𝓝 a ↔ ∃ b, a ∈ F.F b ∧ F.F b ⊆ s := by
  /-
    α : Type u_1
    T : TopologicalSpace α
    F : Ctop.Realizer α
    s : Set α
    a : α
    ⊢ Iff (Membership.mem (nhds a) s) (Exists fun b => And (Membership.mem (F.F.f  …
  -/
  have := @mem_nhds_toTopsp _ _ F.F s a; rwa [F.eq] at this
                                         /-
                                           🎉 no goals
                                         -/


theorem isOpen_iff [TopologicalSpace α] (F : Realizer α) {s : Set α} :
    IsOpen s ↔ ∀ a ∈ s, ∃ b, a ∈ F.F b ∧ F.F b ⊆ s :=
  isOpen_iff_mem_nhds.trans <| forall₂_congr fun _a _h ↦ F.mem_nhds


theorem isClosed_iff [TopologicalSpace α] (F : Realizer α) {s : Set α} :
    IsClosed s ↔ ∀ a, (∀ b, a ∈ F.F b → ∃ z, z ∈ F.F b ∩ s) → a ∈ s :=
  isOpen_compl_iff.symm.trans <|
    F.isOpen_iff.trans <|
      forall_congr' fun a ↦
        show (a ∉ s → ∃ b : F.σ, a ∈ F.F b ∧ ∀ z ∈ F.F b, z ∉ s) ↔ _ by
          /-
            α : Type u_1
            inst✝ : TopologicalSpace α
            F : Ctop.Realizer α
            s : Set α
            a : α
            ⊢ Iff (Not (Membership.mem s a) → Exists fun b => And (Membership.mem (F.F.f b …
          -/
          haveI := Classical.propDecidable; rw [not_imp_comm]
          /-
            α : Type u_1
            inst✝ : TopologicalSpace α
            F : Ctop.Realizer α
            s : Set α
            a : α
            this : (a : Prop) → Decidable a
            ⊢ Iff (Not (Exists fun b => And (Membership.mem (F.F.f b) a) (∀ (z : α), Membe …
          -/
          simp [not_exists, not_and, not_forall, and_comm]
          /-
            🎉 no goals
          -/


theorem mem_interior_iff [TopologicalSpace α] (F : Realizer α) {s : Set α} {a : α} :
    a ∈ interior s ↔ ∃ b, a ∈ F.F b ∧ F.F b ⊆ s :=
  mem_interior_iff_mem_nhds.trans F.mem_nhds


protected theorem isOpen [TopologicalSpace α] (F : Realizer α) (s : F.σ) : IsOpen (F.F s) :=
                                 /-
                                   α : Type u_1
                                   inst✝ : TopologicalSpace α
                                   F : Ctop.Realizer α
                                   s : F.σ
                                   a : α
                                   m : Membership.mem (F.F.f s) a
                                   ⊢ LE.le (nhds a) (Filter.principal (F.F.f s))
                                 -/
  isOpen_iff_nhds.2 fun a m ↦ by simpa using F.mem_nhds.2 ⟨s, m, Subset.refl _⟩
                                 /-
                                   🎉 no goals
                                 -/


theorem ext' [T : TopologicalSpace α] {σ : Type*} {F : Ctop α σ}
    (H : ∀ a s, s ∈ 𝓝 a ↔ ∃ b, a ∈ F b ∧ F b ⊆ s) : F.toTopsp = T := by
  /-
    α : Type u_1
    T : TopologicalSpace α
    σ : Type u_5
    F : Ctop α σ
    H : ∀ (a : α) (s : Set α), Iff (Membership.mem (nhds a) s) (Exists fun b => An …
    ⊢ Eq F.toTopsp T
  -/
  refine TopologicalSpace.ext_nhds fun x ↦ ?_
  /-
    α : Type u_1
    T : TopologicalSpace α
    σ : Type u_5
    F : Ctop α σ
    H : ∀ (a : α) (s : Set α), Iff (Membership.mem (nhds a) s) (Exists fun b => An …
    x : α
    ⊢ Eq (nhds x) (nhds x)
  -/
  ext s
  /-
    case h
    α : Type u_1
    T : TopologicalSpace α
    σ : Type u_5
    F : Ctop α σ
    H : ∀ (a : α) (s : Set α), Iff (Membership.mem (nhds a) s) (Exists fun b => An …
    x : α
    s : Set α
    ⊢ Iff (Membership.mem (nhds x) s) (Membership.mem (nhds x) s)
  -/
  rw [mem_nhds_toTopsp, H]
  /-
    🎉 no goals
  -/


theorem ext [T : TopologicalSpace α] {σ : Type*} {F : Ctop α σ} (H₁ : ∀ a, IsOpen (F a))
    (H₂ : ∀ a s, s ∈ 𝓝 a → ∃ b, a ∈ F b ∧ F b ⊆ s) : F.toTopsp = T :=
  ext' fun a s ↦ ⟨H₂ a s, fun ⟨_b, h₁, h₂⟩ ↦ mem_nhds_iff.2 ⟨_, h₂, H₁ _, h₁⟩⟩


/-- The topological space realizer made of the open sets. -/
protected def id : Realizer α :=
  ⟨{ x : Set α // IsOpen x },
    { f := Subtype.val
      top := fun _ ↦ ⟨univ, isOpen_univ⟩
      top_mem := mem_univ
      inter := fun ⟨_x, h₁⟩ ⟨_y, h₂⟩ _a _h₃ ↦ ⟨_, h₁.inter h₂⟩
      inter_mem := fun ⟨_x, _h₁⟩ ⟨_y, _h₂⟩ _a ↦ id
      inter_sub := fun ⟨_x, _h₁⟩ ⟨_y, _h₂⟩ _a _h₃ ↦ Subset.refl _ },
    ext Subtype.property fun _x _s h ↦
      let ⟨t, h, o, m⟩ := mem_nhds_iff.1 h
      ⟨⟨t, o⟩, m, h⟩⟩


/-- Replace the representation type of a `Ctop` realizer. -/
def ofEquiv (F : Realizer α) (E : F.σ ≃ τ) : Realizer α :=
  ⟨τ, F.F.ofEquiv E,
    ext' fun a s ↦
      F.mem_nhds.trans <|
                               /-
                                 α : Type u_1
                                 β : Type u_2
                                 σ : Type u_3
                                 τ : Type u_4
                                 inst✝ : TopologicalSpace α
                                 F : Ctop.Realizer α
                                 E : Equiv F.σ τ
                                 a : α
                                 s✝ : Set α
                                 x✝ : Exists fun b => And (Membership.mem (F.F.f b) a) (HasSubset.Subset (F.F.f …
                                 s : F.σ
                                 h : And (Membership.mem (F.F.f s) a) (HasSubset.Subset (F.F.f s) s✝)
                                 ⊢ And (Membership.mem ((Ctop.ofEquiv E F.F).f (E s)) a) (HasSubset.Subset ((Ct …
                               -/
                               /-
                                 🎉 no goals
                               -/
        ⟨fun ⟨s, h⟩ ↦ ⟨E s, by simpa using h⟩, fun ⟨t, h⟩ ↦ ⟨E.symm t, by simpa using h⟩⟩⟩
                                                                          /-
                                                                            🎉 no goals
                                                                          -/


@[simp]
theorem ofEquiv_σ (F : Realizer α) (E : F.σ ≃ τ) : (F.ofEquiv E).σ = τ := rfl


@[simp]
theorem ofEquiv_F (F : Realizer α) (E : F.σ ≃ τ) (s : τ) : (F.ofEquiv E).F s = F.F (E.symm s) := by
  /-
    α : Type u_1
    τ : Type u_4
    inst✝ : TopologicalSpace α
    F : Ctop.Realizer α
    E : Equiv F.σ τ
    s : τ
    ⊢ Eq ((F.ofEquiv E).F.f s) (F.F.f (E.symm s))
  -/
  delta ofEquiv; simp
                 /-
                   🎉 no goals
                 -/


/-- A realizer of the neighborhood of a point. -/
protected def nhds (F : Realizer α) (a : α) : (𝓝 a).Realizer :=
  ⟨{ s : F.σ // a ∈ F.F s },
    { f := fun s ↦ F.F s.1
      pt := ⟨_, F.F.top_mem a⟩
      inf := fun ⟨x, h₁⟩ ⟨y, h₂⟩ ↦ ⟨_, F.F.inter_mem x y a ⟨h₁, h₂⟩⟩
      inf_le_left := fun ⟨x, h₁⟩ ⟨y, h₂⟩ _z h ↦ (F.F.inter_sub x y a ⟨h₁, h₂⟩ h).1
      inf_le_right := fun ⟨x, h₁⟩ ⟨y, h₂⟩ _z h ↦ (F.F.inter_sub x y a ⟨h₁, h₂⟩ h).2 },
    filter_eq <|
      Set.ext fun _x ↦
        ⟨fun ⟨⟨_s, as⟩, h⟩ ↦ mem_nhds_iff.2 ⟨_, h, F.isOpen _, as⟩, fun h ↦
          let ⟨s, h, as⟩ := F.mem_nhds.1 h
          ⟨⟨s, h⟩, as⟩⟩⟩


@[simp]
theorem nhds_σ (F : Realizer α) (a : α) : (F.nhds a).σ = { s : F.σ // a ∈ F.F s } := rfl


@[simp]
theorem nhds_F (F : Realizer α) (a : α) (s) : (F.nhds a).F s = F.F s.1 := rfl


theorem tendsto_nhds_iff {m : β → α} {f : Filter β} (F : f.Realizer) (R : Realizer α) {a : α} :
    Tendsto m f (𝓝 a) ↔ ∀ t, a ∈ R.F t → ∃ s, ∀ x ∈ F.F s, m x ∈ R.F t :=
  (F.tendsto_iff _ (R.nhds a)).trans Subtype.forall


/-- A `LocallyFinite.Realizer F f` is a realization that `f` is locally finite, namely it is a
choice of open sets from the basis of `F` such that they intersect only finitely many of the values
of `f`. -/
structure LocallyFinite.Realizer [TopologicalSpace α] (F : Ctop.Realizer α) (f : β → Set α) where
  bas : ∀ a, { s // a ∈ F.F s }
  sets : ∀ x : α, Fintype { i | (f i ∩ F.F (bas x)).Nonempty }


theorem LocallyFinite.Realizer.to_locallyFinite [TopologicalSpace α] {F : Ctop.Realizer α}
    {f : β → Set α} (R : LocallyFinite.Realizer F f) : LocallyFinite f := fun a ↦
  ⟨_, F.mem_nhds.2 ⟨(R.bas a).1, (R.bas a).2, Subset.rfl⟩, have := R.sets a; Set.toFinite _⟩


theorem locallyFinite_iff_exists_realizer [TopologicalSpace α] (F : Ctop.Realizer α)
    {f : β → Set α} : LocallyFinite f ↔ Nonempty (LocallyFinite.Realizer F f) :=
  ⟨fun h ↦
    let ⟨g, h₁⟩ := Classical.axiom_of_choice h
    let ⟨g₂, h₂⟩ :=
      Classical.axiom_of_choice fun x ↦
        show ∃ b : F.σ, x ∈ F.F b ∧ F.F b ⊆ g x from
          let ⟨h, _h'⟩ := h₁ x
          F.mem_nhds.1 h
    ⟨⟨fun x ↦ ⟨g₂ x, (h₂ x).1⟩, fun x ↦
        Finite.fintype <|
          let ⟨_h, h'⟩ := h₁ x
          h'.subset fun _i hi ↦ hi.mono (inter_subset_inter_right _ (h₂ x).2)⟩⟩,
    fun ⟨R⟩ ↦ R.to_locallyFinite⟩


instance [TopologicalSpace α] [Finite β] (F : Ctop.Realizer α) (f : β → Set α) :
    Nonempty (LocallyFinite.Realizer F f) :=
  (locallyFinite_iff_exists_realizer _).1 <| locallyFinite_of_finite _


/-- A `Compact.Realizer s` is a realization that `s` is compact, namely it is a
choice of finite open covers for each set family covering `s`. -/
def Compact.Realizer [TopologicalSpace α] (s : Set α) :=
  ∀ {f : Filter α} (F : f.Realizer) (x : F.σ), f ≠ ⊥ → F.F x ⊆ s → { a // a ∈ s ∧ 𝓝 a ⊓ f ≠ ⊥ }


instance [TopologicalSpace α] : Inhabited (Compact.Realizer (∅ : Set α)) :=
  ⟨fun {f} F x h hF ↦ by
    /-
      α : Type u_1
      β : Type u_2
      σ : Type u_3
      τ : Type u_4
      inst✝ : TopologicalSpace α
      f : Filter α
      F : f.Realizer
      x : F.σ
      h : Ne f Bot.bot
      hF : HasSubset.Subset (F.F.f x) EmptyCollection.emptyCollection
      ⊢ Subtype fun a => And (Membership.mem EmptyCollection.emptyCollection a) (Ne  …
    -/
    suffices f = ⊥ from absurd this h
    /-
      α : Type u_1
      β : Type u_2
      σ : Type u_3
      τ : Type u_4
      inst✝ : TopologicalSpace α
      f : Filter α
      F : f.Realizer
      x : F.σ
      h : Ne f Bot.bot
      hF : HasSubset.Subset (F.F.f x) EmptyCollection.emptyCollection
      ⊢ Eq f Bot.bot
    -/
    rw [← F.eq, eq_bot_iff]
    /-
      α : Type u_1
      β : Type u_2
      σ : Type u_3
      τ : Type u_4
      inst✝ : TopologicalSpace α
      f : Filter α
      F : f.Realizer
      x : F.σ
      h : Ne f Bot.bot
      hF : HasSubset.Subset (F.F.f x) EmptyCollection.emptyCollection
      ⊢ LE.le F.F.toFilter Bot.bot
    -/
    exact fun s _ ↦ ⟨x, hF.trans s.empty_subset⟩⟩
    /-
      🎉 no goals
    -/

