/-- The type of closed subsets of a topological space. -/
structure Closeds (α : Type*) [TopologicalSpace α] where
  /-- the carrier set, i.e. the points in this set -/
  carrier : Set α
  closed' : IsClosed carrier


instance : SetLike (Closeds α) α where
  coe := Closeds.carrier
                             /-
                               ι : Type u_1
                               α : Type u_2
                               β : Type u_3
                               inst✝¹ : TopologicalSpace α
                               inst✝ : TopologicalSpace β
                               s t : TopologicalSpace.Closeds α
                               h : Eq s.carrier t.carrier
                               ⊢ Eq s t
                             -/
  coe_injective' s t h := by cases s; cases t; congr
                                               /-
                                                 🎉 no goals
                                               -/


instance : CanLift (Set α) (Closeds α) (↑) IsClosed where
  prf s hs := ⟨⟨s, hs⟩, rfl⟩


theorem closed (s : Closeds α) : IsClosed (s : Set α) :=
  s.closed'


/-- See Note [custom simps projection]. -/
def Simps.coe (s : Closeds α) : Set α := s


@[ext]
protected theorem ext {s t : Closeds α} (h : (s : Set α) = t) : s = t :=
  SetLike.ext' h


@[simp]
theorem coe_mk (s : Set α) (h) : (mk s h : Set α) = s :=
  rfl


/-- The closure of a set, as an element of `TopologicalSpace.Closeds`. -/
@[simps]
protected def closure (s : Set α) : Closeds α :=
  ⟨closure s, isClosed_closure⟩


@[simp]
theorem mem_closure {s : Set α} {x : α} : x ∈ Closeds.closure s ↔ x ∈ closure s := .rfl


theorem gc : GaloisConnection Closeds.closure ((↑) : Closeds α → Set α) := fun _ U =>
  ⟨subset_closure.trans, fun h => closure_minimal h U.closed⟩


/-- The galois coinsertion between sets and opens. -/
def gi : GaloisInsertion (@Closeds.closure α _) (↑) where
  choice s hs := ⟨s, closure_eq_iff_isClosed.1 <| hs.antisymm subset_closure⟩
  gc := gc
  le_l_u _ := subset_closure
  choice_eq _s hs := SetLike.coe_injective <| subset_closure.antisymm hs


instance instCompleteLattice : CompleteLattice (Closeds α) :=
  CompleteLattice.copy
    (GaloisInsertion.liftCompleteLattice gi)
    -- le
    _ rfl
    -- top
    ⟨univ, isClosed_univ⟩ rfl
    -- bot
    ⟨∅, isClosed_empty⟩ (SetLike.coe_injective closure_empty.symm)
    -- sup
    (fun s t => ⟨s ∪ t, s.2.union t.2⟩)
    (funext fun s => funext fun t => SetLike.coe_injective (s.2.union t.2).closure_eq.symm)
    -- inf
    (fun s t => ⟨s ∩ t, s.2.inter t.2⟩) rfl
    -- sSup
    _ rfl
    -- sInf
    (fun S => ⟨⋂ s ∈ S, ↑s, isClosed_biInter fun s _ => s.2⟩)
    (funext fun _ => SetLike.coe_injective sInf_image.symm)


/-- The type of closed sets is inhabited, with default element the empty set. -/
instance : Inhabited (Closeds α) :=
  ⟨⊥⟩


@[simp, norm_cast]
theorem coe_sup (s t : Closeds α) : (↑(s ⊔ t) : Set α) = ↑s ∪ ↑t := by
  /-
    α : Type u_2
    inst✝ : TopologicalSpace α
    s t : TopologicalSpace.Closeds α
    ⊢ Eq (↑(Max.max s t)) (Union.union ↑s ↑t)
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp, norm_cast]
theorem coe_inf (s t : Closeds α) : (↑(s ⊓ t) : Set α) = ↑s ∩ ↑t :=
  rfl


@[simp, norm_cast]
theorem coe_top : (↑(⊤ : Closeds α) : Set α) = univ :=
  rfl


@[simp, norm_cast]
theorem coe_eq_univ {s : Closeds α} : (s : Set α) = univ ↔ s = ⊤ :=
  SetLike.coe_injective.eq_iff' rfl


@[simp, norm_cast]
theorem coe_bot : (↑(⊥ : Closeds α) : Set α) = ∅ :=
  rfl


@[simp, norm_cast]
theorem coe_eq_empty {s : Closeds α} : (s : Set α) = ∅ ↔ s = ⊥ :=
  SetLike.coe_injective.eq_iff' rfl


theorem coe_nonempty {s : Closeds α} : (s : Set α).Nonempty ↔ s ≠ ⊥ :=
  nonempty_iff_ne_empty.trans coe_eq_empty.not


@[simp, norm_cast]
theorem coe_sInf {S : Set (Closeds α)} : (↑(sInf S) : Set α) = ⋂ i ∈ S, ↑i :=
  rfl


@[simp]
lemma coe_sSup {S : Set (Closeds α)} : ((sSup S : Closeds α) : Set α) =
                                  /-
                                    α : Type u_2
                                    inst✝ : TopologicalSpace α
                                    S : Set (TopologicalSpace.Closeds α)
                                    ⊢ Eq (↑(SupSet.sSup S)) (closure (Set.image SetLike.coe S).sUnion)
                                  -/
    closure (⋃₀ ((↑) '' S)) := by rfl
                                  /-
                                    🎉 no goals
                                  -/


@[simp, norm_cast]
theorem coe_finset_sup (f : ι → Closeds α) (s : Finset ι) :
    (↑(s.sup f) : Set α) = s.sup ((↑) ∘ f) :=
  map_finset_sup (⟨⟨(↑), coe_sup⟩, coe_bot⟩ : SupBotHom (Closeds α) (Set α)) _ _


@[simp, norm_cast]
theorem coe_finset_inf (f : ι → Closeds α) (s : Finset ι) :
    (↑(s.inf f) : Set α) = s.inf ((↑) ∘ f) :=
  map_finset_inf (⟨⟨(↑), coe_inf⟩, coe_top⟩ : InfTopHom (Closeds α) (Set α)) _ _

-- Porting note: Lean 3 proofs didn't work as expected, so I reordered lemmas to fix&golf the proofs


@[simp]
theorem mem_sInf {S : Set (Closeds α)} {x : α} : x ∈ sInf S ↔ ∀ s ∈ S, x ∈ s := mem_iInter₂


@[simp]
                                                                                   /-
                                                                                     α : Type u_2
                                                                                     inst✝ : TopologicalSpace α
                                                                                     ι : Sort u_4
                                                                                     x : α
                                                                                     s : ι → TopologicalSpace.Closeds α
                                                                                     ⊢ Iff (Membership.mem (iInf s) x) (∀ (i : ι), Membership.mem (s i) x)
                                                                                   -/
theorem mem_iInf {ι} {x : α} {s : ι → Closeds α} : x ∈ iInf s ↔ ∀ i, x ∈ s i := by simp [iInf]
                                                                                   /-
                                                                                     🎉 no goals
                                                                                   -/


@[simp, norm_cast]
theorem coe_iInf {ι} (s : ι → Closeds α) : ((⨅ i, s i : Closeds α) : Set α) = ⋂ i, s i := by
  /-
    α : Type u_2
    inst✝ : TopologicalSpace α
    ι : Sort u_4
    s : ι → TopologicalSpace.Closeds α
    ⊢ Eq (↑(iInf fun i => s i)) (Set.iInter fun i => ↑(s i))
  -/
  ext; simp
       /-
         🎉 no goals
       -/


theorem iInf_def {ι} (s : ι → Closeds α) :
                                                                  /-
                                                                    α : Type u_2
                                                                    inst✝ : TopologicalSpace α
                                                                    ι : Sort u_4
                                                                    s : ι → TopologicalSpace.Closeds α
                                                                    ⊢ Eq (iInf fun i => s i) { carrier := Set.iInter fun i => ↑(s i), closed' := ⋯ }
                                                                  -/
    ⨅ i, s i = ⟨⋂ i, s i, isClosed_iInter fun i => (s i).2⟩ := by ext1; simp
                                                                        /-
                                                                          🎉 no goals
                                                                        -/


@[simp]
theorem iInf_mk {ι} (s : ι → Set α) (h : ∀ i, IsClosed (s i)) :
    (⨅ i, ⟨s i, h i⟩ : Closeds α) = ⟨⋂ i, s i, isClosed_iInter h⟩ :=
  iInf_def _


/-- Closed sets in a topological space form a coframe. -/
def coframeMinimalAxioms : Coframe.MinimalAxioms (Closeds α) where
  iInf_sup_le_sup_sInf a s :=
                                 /-
                                   ι : Type u_1
                                   α : Type u_2
                                   β : Type u_3
                                   inst✝¹ : TopologicalSpace α
                                   inst✝ : TopologicalSpace β
                                   a : TopologicalSpace.Closeds α
                                   s : Set (TopologicalSpace.Closeds α)
                                   ⊢ Eq ↑(iInf fun b => iInf fun h => Max.max a b) ↑(Max.max a (InfSet.sInf s))
                                 -/
    (SetLike.coe_injective <| by simp only [coe_sup, coe_iInf, coe_sInf, Set.union_iInter₂]).le
                                 /-
                                   🎉 no goals
                                 -/


instance instCoframe : Coframe (Closeds α) := .ofMinimalAxioms coframeMinimalAxioms


/-- The term of `TopologicalSpace.Closeds α` corresponding to a singleton. -/
@[simps]
def singleton [T1Space α] (x : α) : Closeds α :=
  ⟨{x}, isClosed_singleton⟩


@[simp] lemma mem_singleton [T1Space α] {a b : α} : a ∈ singleton b ↔ a = b := Iff.rfl


/-- The complement of a closed set as an open set. -/
@[simps]
def Closeds.compl (s : Closeds α) : Opens α :=
  ⟨sᶜ, s.2.isOpen_compl⟩


/-- The complement of an open set as a closed set. -/
@[simps]
def Opens.compl (s : Opens α) : Closeds α :=
  ⟨sᶜ, s.2.isClosed_compl⟩


nonrec theorem Closeds.compl_compl (s : Closeds α) : s.compl.compl = s :=
  Closeds.ext (compl_compl (s : Set α))


nonrec theorem Opens.compl_compl (s : Opens α) : s.compl.compl = s :=
  Opens.ext (compl_compl (s : Set α))


theorem Closeds.compl_bijective : Function.Bijective (@Closeds.compl α _) :=
  Function.bijective_iff_has_inverse.mpr ⟨Opens.compl, Closeds.compl_compl, Opens.compl_compl⟩


theorem Opens.compl_bijective : Function.Bijective (@Opens.compl α _) :=
  Function.bijective_iff_has_inverse.mpr ⟨Closeds.compl, Opens.compl_compl, Closeds.compl_compl⟩


/-- `TopologicalSpace.Closeds.compl` as an `OrderIso` to the order dual of
`TopologicalSpace.Opens α`. -/
@[simps]
def Closeds.complOrderIso : Closeds α ≃o (Opens α)ᵒᵈ where
  toFun := OrderDual.toDual ∘ Closeds.compl
  invFun := Opens.compl ∘ OrderDual.ofDual
                   /-
                     ι : Type u_1
                     α : Type u_2
                     β : Type u_3
                     inst✝¹ : TopologicalSpace α
                     inst✝ : TopologicalSpace β
                     s : TopologicalSpace.Closeds α
                     ⊢ Eq (Function.comp TopologicalSpace.Opens.compl (⇑OrderDual.ofDual) (Function …
                   -/
  left_inv s := by simp [Closeds.compl_compl]
                   /-
                     🎉 no goals
                   -/
                    /-
                      ι : Type u_1
                      α : Type u_2
                      β : Type u_3
                      inst✝¹ : TopologicalSpace α
                      inst✝ : TopologicalSpace β
                      s : OrderDual (TopologicalSpace.Opens α)
                      ⊢ Eq (Function.comp (⇑OrderDual.toDual) TopologicalSpace.Closeds.compl (Functi …
                    -/
  right_inv s := by simp [Opens.compl_compl]
                    /-
                      🎉 no goals
                    -/
  map_rel_iff' := (@OrderDual.toDual_le_toDual (Opens α)).trans compl_subset_compl


/-- `TopologicalSpace.Opens.compl` as an `OrderIso` to the order dual of
`TopologicalSpace.Closeds α`. -/
@[simps]
def Opens.complOrderIso : Opens α ≃o (Closeds α)ᵒᵈ where
  toFun := OrderDual.toDual ∘ Opens.compl
  invFun := Closeds.compl ∘ OrderDual.ofDual
                   /-
                     ι : Type u_1
                     α : Type u_2
                     β : Type u_3
                     inst✝¹ : TopologicalSpace α
                     inst✝ : TopologicalSpace β
                     s : TopologicalSpace.Opens α
                     ⊢ Eq (Function.comp TopologicalSpace.Closeds.compl (⇑OrderDual.ofDual) (Functi …
                   -/
  left_inv s := by simp [Opens.compl_compl]
                   /-
                     🎉 no goals
                   -/
                    /-
                      ι : Type u_1
                      α : Type u_2
                      β : Type u_3
                      inst✝¹ : TopologicalSpace α
                      inst✝ : TopologicalSpace β
                      s : OrderDual (TopologicalSpace.Closeds α)
                      ⊢ Eq (Function.comp (⇑OrderDual.toDual) TopologicalSpace.Opens.compl (Function …
                    -/
  right_inv s := by simp [Closeds.compl_compl]
                    /-
                      🎉 no goals
                    -/
  map_rel_iff' := (@OrderDual.toDual_le_toDual (Closeds α)).trans compl_subset_compl


lemma Closeds.coe_eq_singleton_of_isAtom [T0Space α] {s : Closeds α} (hs : IsAtom s) :
    ∃ a, (s : Set α) = {a} := by
  /-
    α : Type u_2
    inst✝¹ : TopologicalSpace α
    inst✝ : T0Space α
    s : TopologicalSpace.Closeds α
    hs : IsAtom s
    ⊢ Exists fun a => Eq (↑s) (Singleton.singleton a)
  -/
  refine minimal_nonempty_closed_eq_singleton s.2 (coe_nonempty.2 hs.1) fun t hts ht ht' ↦ ?_
  /-
    α : Type u_2
    inst✝¹ : TopologicalSpace α
    inst✝ : T0Space α
    s : TopologicalSpace.Closeds α
    hs : IsAtom s
    t : Set α
    hts : HasSubset.Subset t ↑s
    ht : t.Nonempty
    ht' : IsClosed t
    ⊢ Eq t ↑s
  -/
  lift t to Closeds α using ht'
  /-
    case intro
    α : Type u_2
    inst✝¹ : TopologicalSpace α
    inst✝ : T0Space α
    s : TopologicalSpace.Closeds α
    hs : IsAtom s
    t : TopologicalSpace.Closeds α
    hts : HasSubset.Subset ↑t ↑s
    ht : (↑t).Nonempty
    ⊢ Eq ↑t ↑s
  -/
  exact SetLike.coe_injective.eq_iff.2 <| (hs.le_iff_eq <| coe_nonempty.1 ht).1 hts
  /-
    🎉 no goals
  -/


@[simp, norm_cast] lemma Closeds.isAtom_coe [T1Space α] {s : Closeds α} :
    IsAtom (s : Set α) ↔ IsAtom s :=
  Closeds.gi.isAtom_iff' rfl
                   /-
                     α : Type u_2
                     inst✝¹ : TopologicalSpace α
                     inst✝ : T1Space α
                     s : TopologicalSpace.Closeds α
                     t : Set α
                     ht : IsAtom t
                     ⊢ Eq (↑(TopologicalSpace.Closeds.closure t)) t
                   -/
    (fun t ht ↦ by obtain ⟨x, rfl⟩ := Set.isAtom_iff.1 ht; exact closure_singleton) s
                                                           /-
                                                             🎉 no goals
                                                           -/


/-- in a `T1Space`, atoms of `TopologicalSpace.Closeds α` are precisely the
`TopologicalSpace.Closeds.singleton`s. -/
theorem Closeds.isAtom_iff [T1Space α] {s : Closeds α} :
    IsAtom s ↔ ∃ x, s = Closeds.singleton x := by
  /-
    α : Type u_2
    inst✝¹ : TopologicalSpace α
    inst✝ : T1Space α
    s : TopologicalSpace.Closeds α
    ⊢ Iff (IsAtom s) (Exists fun x => Eq s (TopologicalSpace.Closeds.singleton x))
  -/
  simp [← Closeds.isAtom_coe, Set.isAtom_iff, SetLike.ext_iff, Set.ext_iff]
  /-
    🎉 no goals
  -/


/-- in a `T1Space`, coatoms of `TopologicalSpace.Opens α` are precisely complements of singletons:
`(TopologicalSpace.Closeds.singleton x).compl`. -/
theorem Opens.isCoatom_iff [T1Space α] {s : Opens α} :
    IsCoatom s ↔ ∃ x, s = (Closeds.singleton x).compl := by
  /-
    α : Type u_2
    inst✝¹ : TopologicalSpace α
    inst✝ : T1Space α
    s : TopologicalSpace.Opens α
    ⊢ Iff (IsCoatom s) (Exists fun x => Eq s (TopologicalSpace.Closeds.singleton x …
  -/
  rw [← s.compl_compl, ← isAtom_dual_iff_isCoatom]
  /-
    α : Type u_2
    inst✝¹ : TopologicalSpace α
    inst✝ : T1Space α
    s : TopologicalSpace.Opens α
    ⊢ Iff (IsAtom (OrderDual.toDual s.compl.compl)) (Exists fun x => Eq s.compl.co …
  -/
  change IsAtom (Closeds.complOrderIso α s.compl) ↔ _
  simp only [(Closeds.complOrderIso α).isAtom_iff, Closeds.isAtom_iff,
    Closeds.compl_bijective.injective.eq_iff]


/-- The type of clopen sets of a topological space. -/
structure Clopens (α : Type*) [TopologicalSpace α] where
  /-- the carrier set, i.e. the points in this set -/
  carrier : Set α
  isClopen' : IsClopen carrier


instance : SetLike (Clopens α) α where
  coe s := s.carrier
                             /-
                               ι : Type u_1
                               α : Type u_2
                               β : Type u_3
                               inst✝¹ : TopologicalSpace α
                               inst✝ : TopologicalSpace β
                               s t : TopologicalSpace.Clopens α
                               h : Eq ((fun s => s.carrier) s) ((fun s => s.carrier) t)
                               ⊢ Eq s t
                             -/
  coe_injective' s t h := by cases s; cases t; congr
                                               /-
                                                 🎉 no goals
                                               -/


theorem isClopen (s : Clopens α) : IsClopen (s : Set α) :=
  s.isClopen'


/-- See Note [custom simps projection]. -/
def Simps.coe (s : Clopens α) : Set α := s


/-- Reinterpret a clopen as an open. -/
@[simps]
def toOpens (s : Clopens α) : Opens α :=
  ⟨s, s.isClopen.isOpen⟩


@[ext]
protected theorem ext {s t : Clopens α} (h : (s : Set α) = t) : s = t :=
  SetLike.ext' h


@[simp] lemma mem_mk {s : Set α} {x h} : x ∈ mk s h ↔ x ∈ s := .rfl


instance : Max (Clopens α) := ⟨fun s t => ⟨s ∪ t, s.isClopen.union t.isClopen⟩⟩

instance : Min (Clopens α) := ⟨fun s t => ⟨s ∩ t, s.isClopen.inter t.isClopen⟩⟩

instance : Top (Clopens α) := ⟨⟨⊤, isClopen_univ⟩⟩

instance : Bot (Clopens α) := ⟨⟨⊥, isClopen_empty⟩⟩

instance : SDiff (Clopens α) := ⟨fun s t => ⟨s \ t, s.isClopen.diff t.isClopen⟩⟩

instance : HImp (Clopens α) where himp s t := ⟨s ⇨ t, s.isClopen.himp t.isClopen⟩

instance : HasCompl (Clopens α) := ⟨fun s => ⟨sᶜ, s.isClopen.compl⟩⟩


@[simp, norm_cast] lemma coe_sup (s t : Clopens α) : ↑(s ⊔ t) = (s ∪ t : Set α) := rfl

@[simp, norm_cast] lemma coe_inf (s t : Clopens α) : ↑(s ⊓ t) = (s ∩ t : Set α) := rfl

@[simp, norm_cast] lemma coe_top : (↑(⊤ : Clopens α) : Set α) = univ := rfl

@[simp, norm_cast] lemma coe_bot : (↑(⊥ : Clopens α) : Set α) = ∅ := rfl

@[simp, norm_cast] lemma coe_sdiff (s t : Clopens α) : ↑(s \ t) = (s \ t : Set α) := rfl

@[simp, norm_cast] lemma coe_himp (s t : Clopens α) : ↑(s ⇨ t) = (s ⇨ t : Set α) := rfl

@[simp, norm_cast] lemma coe_compl (s : Clopens α) : (↑sᶜ : Set α) = (↑s)ᶜ := rfl


instance : BooleanAlgebra (Clopens α) :=
  SetLike.coe_injective.booleanAlgebra _ coe_sup coe_inf coe_top coe_bot coe_compl coe_sdiff
    coe_himp


instance : Inhabited (Clopens α) := ⟨⊥⟩


instance : SProd (Clopens α) (Clopens β) (Clopens (α × β)) where
  sprod s t := ⟨s ×ˢ t, s.2.prod t.2⟩


@[simp]
protected lemma mem_prod {s : Clopens α} {t : Clopens β} {x : α × β} :
    x ∈ s ×ˢ t ↔ x.1 ∈ s ∧ x.2 ∈ t := .rfl


/-- The type of irreducible closed subsets of a topological space. -/
structure IrreducibleCloseds (α : Type*) [TopologicalSpace α] where
  /-- the carrier set, i.e. the points in this set -/
  carrier : Set α
  is_irreducible' : IsIrreducible carrier
  is_closed' : IsClosed carrier


instance : SetLike (IrreducibleCloseds α) α where
  coe := IrreducibleCloseds.carrier
                             /-
                               ι : Type u_1
                               α : Type u_2
                               β : Type u_3
                               inst✝¹ : TopologicalSpace α
                               inst✝ : TopologicalSpace β
                               s t : TopologicalSpace.IrreducibleCloseds α
                               h : Eq s.carrier t.carrier
                               ⊢ Eq s t
                             -/
  coe_injective' s t h := by cases s; cases t; congr
                                               /-
                                                 🎉 no goals
                                               -/


instance : CanLift (Set α) (IrreducibleCloseds α) (↑) (fun s ↦ IsIrreducible s ∧ IsClosed s) where
  prf s hs := ⟨⟨s, hs.1, hs.2⟩, rfl⟩


theorem isIrreducible (s : IrreducibleCloseds α) : IsIrreducible (s : Set α) := s.is_irreducible'


theorem isClosed (s : IrreducibleCloseds α) : IsClosed (s : Set α) := s.is_closed'


/-- See Note [custom simps projection]. -/
def Simps.coe (s : IrreducibleCloseds α) : Set α := s


@[ext]
protected theorem ext {s t : IrreducibleCloseds α} (h : (s : Set α) = t) : s = t :=
  SetLike.ext' h


@[simp]
theorem coe_mk (s : Set α) (h : IsIrreducible s) (h' : IsClosed s) : (mk s h h' : Set α) = s :=
  rfl


/-- The term of `TopologicalSpace.IrreducibleCloseds α` corresponding to a singleton. -/
@[simps]
def singleton [T1Space α] (x : α) : IrreducibleCloseds α :=
  ⟨{x}, isIrreducible_singleton, isClosed_singleton⟩


/--
The equivalence between `IrreducibleCloseds α` and `{x : Set α // IsIrreducible x ∧ IsClosed x }`.
-/
@[simps apply symm_apply]
def equivSubtype : IrreducibleCloseds α ≃ { x : Set α // IsIrreducible x ∧ IsClosed x } where
  toFun a   := ⟨a.1, a.2, a.3⟩
  invFun a  := ⟨a.1, a.2.1, a.2.2⟩
  left_inv  := fun ⟨_, _, _⟩ => rfl
  right_inv := fun ⟨_, _, _⟩ => rfl


/--
The equivalence between `IrreducibleCloseds α` and `{x : Set α // IsClosed x ∧ IsIrreducible x }`.
-/
@[simps apply symm_apply]
def equivSubtype' : IrreducibleCloseds α ≃ { x : Set α // IsClosed x ∧ IsIrreducible x } where
  toFun a   := ⟨a.1, a.3, a.2⟩
  invFun a  := ⟨a.1, a.2.2, a.2.1⟩
  left_inv  := fun ⟨_, _, _⟩ => rfl
  right_inv := fun ⟨_, _, _⟩ => rfl


variable (α) in
/-- The equivalence `IrreducibleCloseds α ≃ { x : Set α // IsIrreducible x ∧ IsClosed x }` is an
order isomorphism.-/
def orderIsoSubtype : IrreducibleCloseds α ≃o { x : Set α // IsIrreducible x ∧ IsClosed x } :=
  equivSubtype.toOrderIso (fun _ _ h ↦ h) (fun _ _ h ↦ h)


variable (α) in
/-- The equivalence `IrreducibleCloseds α ≃ { x : Set α // IsClosed x ∧ IsIrreducible x }` is an
order isomorphism.-/
def orderIsoSubtype' : IrreducibleCloseds α ≃o { x : Set α // IsClosed x ∧ IsIrreducible x } :=
  equivSubtype'.toOrderIso (fun _ _ h ↦ h) (fun _ _ h ↦ h)


