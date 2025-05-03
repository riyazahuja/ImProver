/-- The graph of a finitely supported function over its support, i.e. the finset of input and output
pairs with non-zero outputs. -/
def graph (f : α →₀ M) : Finset (α × M) :=
  f.support.map ⟨fun a => Prod.mk a (f a), fun _ _ h => (Prod.mk.inj h).1⟩


theorem mk_mem_graph_iff {a : α} {m : M} {f : α →₀ M} : (a, m) ∈ f.graph ↔ f a = m ∧ m ≠ 0 := by
  /-
    α : Type u_1
    M : Type u_5
    inst✝ : Zero M
    a : α
    m : M
    f : Finsupp α M
    ⊢ Iff (Membership.mem f.graph { fst := a, snd := m }) (And (Eq (f a) m) (Ne m  …
  -/
  simp_rw [graph, mem_map, mem_support_iff]
  /-
    α : Type u_1
    M : Type u_5
    inst✝ : Zero M
    a : α
    m : M
    f : Finsupp α M
    ⊢ Iff (Exists fun a_1 => And (Ne (f a_1) 0) (Eq ({ toFun := fun a => { fst :=  …
  -/
  constructor
    /-
      case mp
      α : Type u_1
      M : Type u_5
      inst✝ : Zero M
      a : α
      m : M
      f : Finsupp α M
      ⊢ (Exists fun a_1 => And (Ne (f a_1) 0) (Eq ({ toFun := fun a => { fst := a, s …
    -/
  · rintro ⟨b, ha, rfl, -⟩
    /-
      case mp.intro.intro.refl
      α : Type u_1
      M : Type u_5
      inst✝ : Zero M
      a : α
      f : Finsupp α M
      ha : Ne (f a) 0
      ⊢ And (Eq (f a) (f a)) (Ne (f a) 0)
    -/
    exact ⟨rfl, ha⟩
    /-
      🎉 no goals
    -/
    /-
      case mpr
      α : Type u_1
      M : Type u_5
      inst✝ : Zero M
      a : α
      m : M
      f : Finsupp α M
      ⊢ And (Eq (f a) m) (Ne m 0) → Exists fun a_2 => And (Ne (f a_2) 0) (Eq ({ toFu …
    -/
  · rintro ⟨rfl, ha⟩
    /-
      case mpr.intro
      α : Type u_1
      M : Type u_5
      inst✝ : Zero M
      a : α
      f : Finsupp α M
      ha : Ne (f a) 0
      ⊢ Exists fun a_1 => And (Ne (f a_1) 0) (Eq ({ toFun := fun a => { fst := a, sn …
    -/
    exact ⟨a, ha, rfl⟩
    /-
      🎉 no goals
    -/


@[simp]
theorem mem_graph_iff {c : α × M} {f : α →₀ M} : c ∈ f.graph ↔ f c.1 = c.2 ∧ c.2 ≠ 0 := by
  /-
    α : Type u_1
    M : Type u_5
    inst✝ : Zero M
    c : Prod α M
    f : Finsupp α M
    ⊢ Iff (Membership.mem f.graph c) (And (Eq (f c.1) c.2) (Ne c.2 0))
  -/
  cases c
  /-
    case mk
    α : Type u_1
    M : Type u_5
    inst✝ : Zero M
    f : Finsupp α M
    fst✝ : α
    snd✝ : M
    ⊢ Iff (Membership.mem f.graph { fst := fst✝, snd := snd✝ }) (And (Eq (f { fst  …
  -/
  exact mk_mem_graph_iff
  /-
    🎉 no goals
  -/


theorem mk_mem_graph (f : α →₀ M) {a : α} (ha : a ∈ f.support) : (a, f a) ∈ f.graph :=
  mk_mem_graph_iff.2 ⟨rfl, mem_support_iff.1 ha⟩


theorem apply_eq_of_mem_graph {a : α} {m : M} {f : α →₀ M} (h : (a, m) ∈ f.graph) : f a = m :=
  (mem_graph_iff.1 h).1


@[simp 1100] -- Porting note: change priority to appease `simpNF`
theorem not_mem_graph_snd_zero (a : α) (f : α →₀ M) : (a, (0 : M)) ∉ f.graph := fun h =>
  (mem_graph_iff.1 h).2.irrefl


@[simp]
theorem image_fst_graph [DecidableEq α] (f : α →₀ M) : f.graph.image Prod.fst = f.support := by
  classical
  simp only [graph, map_eq_image, image_image, Embedding.coeFn_mk, Function.comp_def, image_id']


theorem graph_injective (α M) [Zero M] : Injective (@graph α M _) := by
  /-
    α : Type u_13
    M : Type u_14
    inst✝ : Zero M
    ⊢ Function.Injective Finsupp.graph
  -/
  intro f g h
  classical
    have hsup : f.support = g.support := by rw [← image_fst_graph, h, image_fst_graph]
    refine ext_iff'.2 ⟨hsup, fun x hx => apply_eq_of_mem_graph <| h.symm ▸ ?_⟩
    exact mk_mem_graph _ (hsup ▸ hx)


@[simp]
theorem graph_inj {f g : α →₀ M} : f.graph = g.graph ↔ f = g :=
  (graph_injective α M).eq_iff


@[simp]
                                                  /-
                                                    α : Type u_1
                                                    M : Type u_5
                                                    inst✝ : Zero M
                                                    ⊢ Eq (Finsupp.graph 0) EmptyCollection.emptyCollection
                                                  -/
theorem graph_zero : graph (0 : α →₀ M) = ∅ := by simp [graph]
                                                  /-
                                                    🎉 no goals
                                                  -/


@[simp]
theorem graph_eq_empty {f : α →₀ M} : f.graph = ∅ ↔ f = 0 :=
  (graph_injective α M).eq_iff' graph_zero


/-- `Finsupp.mapRange` as an equiv. -/
@[simps apply]
def mapRange.equiv (f : M ≃ N) (hf : f 0 = 0) (hf' : f.symm 0 = 0) : (α →₀ M) ≃ (α →₀ N) where
  toFun := (mapRange f hf : (α →₀ M) → α →₀ N)
  invFun := (mapRange f.symm hf' : (α →₀ N) → α →₀ M)
  left_inv x := by
    /-
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      ι : Type u_4
      M : Type u_5
      M' : Type u_6
      N : Type u_7
      P : Type u_8
      G : Type u_9
      H : Type u_10
      R : Type u_11
      S : Type u_12
      inst✝² : Zero M
      inst✝¹ : Zero N
      inst✝ : Zero P
      f : Equiv M N
      hf : Eq (f 0) 0
      hf' : Eq (f.symm 0) 0
      x : Finsupp α M
      ⊢ Eq (Finsupp.mapRange (⇑f.symm) hf' (Finsupp.mapRange (⇑f) hf x)) x
    -/
    rw [← mapRange_comp _ _ _ _] <;> simp_rw [Equiv.symm_comp_self]
      /-
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        ι : Type u_4
        M : Type u_5
        M' : Type u_6
        N : Type u_7
        P : Type u_8
        G : Type u_9
        H : Type u_10
        R : Type u_11
        S : Type u_12
        inst✝² : Zero M
        inst✝¹ : Zero N
        inst✝ : Zero P
        f : Equiv M N
        hf : Eq (f 0) 0
        hf' : Eq (f.symm 0) 0
        x : Finsupp α M
        ⊢ Eq (Finsupp.mapRange id ⋯ x) x
      -/
    · exact mapRange_id _
      /-
        🎉 no goals
      -/
      /-
        case h
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        ι : Type u_4
        M : Type u_5
        M' : Type u_6
        N : Type u_7
        P : Type u_8
        G : Type u_9
        H : Type u_10
        R : Type u_11
        S : Type u_12
        inst✝² : Zero M
        inst✝¹ : Zero N
        inst✝ : Zero P
        f : Equiv M N
        hf : Eq (f 0) 0
        hf' : Eq (f.symm 0) 0
        x : Finsupp α M
        ⊢ Eq (id 0) 0
      -/
    · rfl
      /-
        🎉 no goals
      -/
  right_inv x := by
    /-
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      ι : Type u_4
      M : Type u_5
      M' : Type u_6
      N : Type u_7
      P : Type u_8
      G : Type u_9
      H : Type u_10
      R : Type u_11
      S : Type u_12
      inst✝² : Zero M
      inst✝¹ : Zero N
      inst✝ : Zero P
      f : Equiv M N
      hf : Eq (f 0) 0
      hf' : Eq (f.symm 0) 0
      x : Finsupp α N
      ⊢ Eq (Finsupp.mapRange (⇑f) hf (Finsupp.mapRange (⇑f.symm) hf' x)) x
    -/
    rw [← mapRange_comp _ _ _ _] <;> simp_rw [Equiv.self_comp_symm]
      /-
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        ι : Type u_4
        M : Type u_5
        M' : Type u_6
        N : Type u_7
        P : Type u_8
        G : Type u_9
        H : Type u_10
        R : Type u_11
        S : Type u_12
        inst✝² : Zero M
        inst✝¹ : Zero N
        inst✝ : Zero P
        f : Equiv M N
        hf : Eq (f 0) 0
        hf' : Eq (f.symm 0) 0
        x : Finsupp α N
        ⊢ Eq (Finsupp.mapRange id ⋯ x) x
      -/
    · exact mapRange_id _
      /-
        🎉 no goals
      -/
      /-
        case h
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        ι : Type u_4
        M : Type u_5
        M' : Type u_6
        N : Type u_7
        P : Type u_8
        G : Type u_9
        H : Type u_10
        R : Type u_11
        S : Type u_12
        inst✝² : Zero M
        inst✝¹ : Zero N
        inst✝ : Zero P
        f : Equiv M N
        hf : Eq (f 0) 0
        hf' : Eq (f.symm 0) 0
        x : Finsupp α N
        ⊢ Eq (id 0) 0
      -/
    · rfl
      /-
        🎉 no goals
      -/


@[simp]
theorem mapRange.equiv_refl : mapRange.equiv (Equiv.refl M) rfl rfl = Equiv.refl (α →₀ M) :=
  Equiv.ext mapRange_id


theorem mapRange.equiv_trans (f : M ≃ N) (hf : f 0 = 0) (hf') (f₂ : N ≃ P) (hf₂ : f₂ 0 = 0) (hf₂') :
                                     /-
                                       α : Type u_1
                                       β : Type u_2
                                       γ : Type u_3
                                       ι : Type u_4
                                       M : Type u_5
                                       M' : Type u_6
                                       N : Type u_7
                                       P : Type u_8
                                       G : Type u_9
                                       H : Type u_10
                                       R : Type u_11
                                       S : Type u_12
                                       inst✝² : Zero M
                                       inst✝¹ : Zero N
                                       inst✝ : Zero P
                                       f : Equiv M N
                                       hf : Eq (f 0) 0
                                       hf' : Eq (f.symm 0) 0
                                       f₂ : Equiv N P
                                       hf₂ : Eq (f₂ 0) 0
                                       hf₂' : Eq (f₂.symm 0) 0
                                       ⊢ Eq ((f.trans f₂) 0) 0
                                     -/
    (mapRange.equiv (f.trans f₂) (by rw [Equiv.trans_apply, hf, hf₂])
                                     /-
                                       🎉 no goals
                                     -/
              /-
                α : Type u_1
                β : Type u_2
                γ : Type u_3
                ι : Type u_4
                M : Type u_5
                M' : Type u_6
                N : Type u_7
                P : Type u_8
                G : Type u_9
                H : Type u_10
                R : Type u_11
                S : Type u_12
                inst✝² : Zero M
                inst✝¹ : Zero N
                inst✝ : Zero P
                f : Equiv M N
                hf : Eq (f 0) 0
                hf' : Eq (f.symm 0) 0
                f₂ : Equiv N P
                hf₂ : Eq (f₂ 0) 0
                hf₂' : Eq (f₂.symm 0) 0
                ⊢ Eq ((f.trans f₂).symm 0) 0
              -/
          (by rw [Equiv.symm_trans_apply, hf₂', hf']) :
              /-
                🎉 no goals
              -/
        (α →₀ _) ≃ _) =
      (mapRange.equiv f hf hf').trans (mapRange.equiv f₂ hf₂ hf₂') :=
  Equiv.ext <| mapRange_comp f₂ hf₂ f hf ((congrArg f₂ hf).trans hf₂)


@[simp]
theorem mapRange.equiv_symm (f : M ≃ N) (hf hf') :
    ((mapRange.equiv f hf hf').symm : (α →₀ _) ≃ _) = mapRange.equiv f.symm hf' hf :=
  Equiv.ext fun _ => rfl


/-- Composition with a fixed zero-preserving homomorphism is itself a zero-preserving homomorphism
on functions. -/
@[simps]
def mapRange.zeroHom (f : ZeroHom M N) : ZeroHom (α →₀ M) (α →₀ N) where
  toFun := (mapRange f f.map_zero : (α →₀ M) → α →₀ N)
  map_zero' := mapRange_zero


@[simp]
theorem mapRange.zeroHom_id : mapRange.zeroHom (ZeroHom.id M) = ZeroHom.id (α →₀ M) :=
  ZeroHom.ext mapRange_id


theorem mapRange.zeroHom_comp (f : ZeroHom N P) (f₂ : ZeroHom M N) :
    (mapRange.zeroHom (f.comp f₂) : ZeroHom (α →₀ _) _) =
      (mapRange.zeroHom f).comp (mapRange.zeroHom f₂) :=
                                                                   /-
                                                                     α : Type u_1
                                                                     M : Type u_5
                                                                     N : Type u_7
                                                                     P : Type u_8
                                                                     inst✝² : Zero M
                                                                     inst✝¹ : Zero N
                                                                     inst✝ : Zero P
                                                                     f : ZeroHom N P
                                                                     f₂ : ZeroHom M N
                                                                     ⊢ Eq (Function.comp (⇑f) (⇑f₂) 0) 0
                                                                   -/
  ZeroHom.ext <| mapRange_comp f (map_zero f) f₂ (map_zero f₂) (by simp only [comp_apply, map_zero])
                                                                   /-
                                                                     🎉 no goals
                                                                   -/


/-- Composition with a fixed additive homomorphism is itself an additive homomorphism on functions.
-/
@[simps]
def mapRange.addMonoidHom (f : M →+ N) : (α →₀ M) →+ α →₀ N where
  toFun := (mapRange f f.map_zero : (α →₀ M) → α →₀ N)
  map_zero' := mapRange_zero
                     /-
                       α : Type u_1
                       β : Type u_2
                       γ : Type u_3
                       ι : Type u_4
                       M : Type u_5
                       M' : Type u_6
                       N : Type u_7
                       P : Type u_8
                       G : Type u_9
                       H : Type u_10
                       R : Type u_11
                       S : Type u_12
                       inst✝⁴ : AddCommMonoid M
                       inst✝³ : AddCommMonoid N
                       inst✝² : AddCommMonoid P
                       F : Type u_13
                       inst✝¹ : FunLike F M N
                       inst✝ : AddMonoidHomClass F M N
                       f : AddMonoidHom M N
                       a b : Finsupp α M
                       ⊢ Eq ({ toFun := Finsupp.mapRange ⇑f ⋯, map_zero' := ⋯ }.toFun (HAdd.hAdd a b) …
                     -/
  map_add' a b := by dsimp only; exact mapRange_add f.map_add _ _; -- Porting note: `dsimp` needed
                                 /-
                                   🎉 no goals
                                 -/


@[simp]
theorem mapRange.addMonoidHom_id :
    mapRange.addMonoidHom (AddMonoidHom.id M) = AddMonoidHom.id (α →₀ M) :=
  AddMonoidHom.ext mapRange_id


theorem mapRange.addMonoidHom_comp (f : N →+ P) (f₂ : M →+ N) :
    (mapRange.addMonoidHom (f.comp f₂) : (α →₀ _) →+ _) =
      (mapRange.addMonoidHom f).comp (mapRange.addMonoidHom f₂) :=
  AddMonoidHom.ext <|
                                                      /-
                                                        α : Type u_1
                                                        M : Type u_5
                                                        N : Type u_7
                                                        P : Type u_8
                                                        inst✝² : AddCommMonoid M
                                                        inst✝¹ : AddCommMonoid N
                                                        inst✝ : AddCommMonoid P
                                                        f : AddMonoidHom N P
                                                        f₂ : AddMonoidHom M N
                                                        ⊢ Eq (Function.comp (⇑f) (⇑f₂) 0) 0
                                                      -/
    mapRange_comp f (map_zero f) f₂ (map_zero f₂) (by simp only [comp_apply, map_zero])
                                                      /-
                                                        🎉 no goals
                                                      -/


@[simp]
theorem mapRange.addMonoidHom_toZeroHom (f : M →+ N) :
    (mapRange.addMonoidHom f).toZeroHom = (mapRange.zeroHom f.toZeroHom : ZeroHom (α →₀ _) _) :=
  ZeroHom.ext fun _ => rfl


theorem mapRange_multiset_sum (f : F) (m : Multiset (α →₀ M)) :
    mapRange f (map_zero f) m.sum = (m.map fun x => mapRange f (map_zero f) x).sum :=
  (mapRange.addMonoidHom (f : M →+ N) : (α →₀ _) →+ _).map_multiset_sum _


theorem mapRange_finset_sum (f : F) (s : Finset ι) (g : ι → α →₀ M) :
    mapRange f (map_zero f) (∑ x ∈ s, g x) = ∑ x ∈ s, mapRange f (map_zero f) (g x) :=
  map_sum (mapRange.addMonoidHom (f : M →+ N)) _ _


/-- `Finsupp.mapRange.AddMonoidHom` as an equiv. -/
@[simps apply]
def mapRange.addEquiv (f : M ≃+ N) : (α →₀ M) ≃+ (α →₀ N) :=
  { mapRange.addMonoidHom f.toAddMonoidHom with
    toFun := (mapRange f f.map_zero : (α →₀ M) → α →₀ N)
    invFun := (mapRange f.symm f.symm.map_zero : (α →₀ N) → α →₀ M)
    left_inv := fun x => by
      /-
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        ι : Type u_4
        M : Type u_5
        M' : Type u_6
        N : Type u_7
        P : Type u_8
        G : Type u_9
        H : Type u_10
        R : Type u_11
        S : Type u_12
        inst✝⁴ : AddCommMonoid M
        inst✝³ : AddCommMonoid N
        inst✝² : AddCommMonoid P
        F : Type u_13
        inst✝¹ : FunLike F M N
        inst✝ : AddMonoidHomClass F M N
        f : AddEquiv M N
        x : Finsupp α M
        ⊢ Eq (Finsupp.mapRange ⇑f.symm ⋯ (Finsupp.mapRange ⇑f ⋯ x)) x
      -/
      rw [← mapRange_comp _ _ _ _] <;> simp_rw [AddEquiv.symm_comp_self]
        /-
          α : Type u_1
          β : Type u_2
          γ : Type u_3
          ι : Type u_4
          M : Type u_5
          M' : Type u_6
          N : Type u_7
          P : Type u_8
          G : Type u_9
          H : Type u_10
          R : Type u_11
          S : Type u_12
          inst✝⁴ : AddCommMonoid M
          inst✝³ : AddCommMonoid N
          inst✝² : AddCommMonoid P
          F : Type u_13
          inst✝¹ : FunLike F M N
          inst✝ : AddMonoidHomClass F M N
          f : AddEquiv M N
          x : Finsupp α M
          ⊢ Eq (Finsupp.mapRange id ⋯ x) x
        -/
      · exact mapRange_id _
        /-
          🎉 no goals
        -/
        /-
          case h
          α : Type u_1
          β : Type u_2
          γ : Type u_3
          ι : Type u_4
          M : Type u_5
          M' : Type u_6
          N : Type u_7
          P : Type u_8
          G : Type u_9
          H : Type u_10
          R : Type u_11
          S : Type u_12
          inst✝⁴ : AddCommMonoid M
          inst✝³ : AddCommMonoid N
          inst✝² : AddCommMonoid P
          F : Type u_13
          inst✝¹ : FunLike F M N
          inst✝ : AddMonoidHomClass F M N
          f : AddEquiv M N
          x : Finsupp α M
          ⊢ Eq (id 0) 0
        -/
      · rfl
        /-
          🎉 no goals
        -/
    right_inv := fun x => by
      /-
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        ι : Type u_4
        M : Type u_5
        M' : Type u_6
        N : Type u_7
        P : Type u_8
        G : Type u_9
        H : Type u_10
        R : Type u_11
        S : Type u_12
        inst✝⁴ : AddCommMonoid M
        inst✝³ : AddCommMonoid N
        inst✝² : AddCommMonoid P
        F : Type u_13
        inst✝¹ : FunLike F M N
        inst✝ : AddMonoidHomClass F M N
        f : AddEquiv M N
        x : Finsupp α N
        ⊢ Eq (Finsupp.mapRange ⇑f ⋯ (Finsupp.mapRange ⇑f.symm ⋯ x)) x
      -/
      rw [← mapRange_comp _ _ _ _] <;> simp_rw [AddEquiv.self_comp_symm]
        /-
          α : Type u_1
          β : Type u_2
          γ : Type u_3
          ι : Type u_4
          M : Type u_5
          M' : Type u_6
          N : Type u_7
          P : Type u_8
          G : Type u_9
          H : Type u_10
          R : Type u_11
          S : Type u_12
          inst✝⁴ : AddCommMonoid M
          inst✝³ : AddCommMonoid N
          inst✝² : AddCommMonoid P
          F : Type u_13
          inst✝¹ : FunLike F M N
          inst✝ : AddMonoidHomClass F M N
          f : AddEquiv M N
          x : Finsupp α N
          ⊢ Eq (Finsupp.mapRange id ⋯ x) x
        -/
      · exact mapRange_id _
        /-
          🎉 no goals
        -/
        /-
          case h
          α : Type u_1
          β : Type u_2
          γ : Type u_3
          ι : Type u_4
          M : Type u_5
          M' : Type u_6
          N : Type u_7
          P : Type u_8
          G : Type u_9
          H : Type u_10
          R : Type u_11
          S : Type u_12
          inst✝⁴ : AddCommMonoid M
          inst✝³ : AddCommMonoid N
          inst✝² : AddCommMonoid P
          F : Type u_13
          inst✝¹ : FunLike F M N
          inst✝ : AddMonoidHomClass F M N
          f : AddEquiv M N
          x : Finsupp α N
          ⊢ Eq (id 0) 0
        -/
      · rfl }
        /-
          🎉 no goals
        -/


@[simp]
theorem mapRange.addEquiv_refl : mapRange.addEquiv (AddEquiv.refl M) = AddEquiv.refl (α →₀ M) :=
  AddEquiv.ext mapRange_id


theorem mapRange.addEquiv_trans (f : M ≃+ N) (f₂ : N ≃+ P) :
    (mapRange.addEquiv (f.trans f₂) : (α →₀ M) ≃+ (α →₀ P)) =
      (mapRange.addEquiv f).trans (mapRange.addEquiv f₂) :=
                                                             /-
                                                               α : Type u_1
                                                               M : Type u_5
                                                               N : Type u_7
                                                               P : Type u_8
                                                               inst✝² : AddCommMonoid M
                                                               inst✝¹ : AddCommMonoid N
                                                               inst✝ : AddCommMonoid P
                                                               f : AddEquiv M N
                                                               f₂ : AddEquiv N P
                                                               ⊢ Eq (Function.comp (⇑f₂) (⇑f) 0) 0
                                                             -/
  AddEquiv.ext (mapRange_comp _ f₂.map_zero _ f.map_zero (by simp))
                                                             /-
                                                               🎉 no goals
                                                             -/


@[simp]
theorem mapRange.addEquiv_symm (f : M ≃+ N) :
    ((mapRange.addEquiv f).symm : (α →₀ _) ≃+ _) = mapRange.addEquiv f.symm :=
  AddEquiv.ext fun _ => rfl


@[simp]
theorem mapRange.addEquiv_toAddMonoidHom (f : M ≃+ N) :
    ((mapRange.addEquiv f : (α →₀ _) ≃+ _) : _ →+ _) =
      (mapRange.addMonoidHom f.toAddMonoidHom : (α →₀ _) →+ _) :=
  AddMonoidHom.ext fun _ => rfl


@[simp]
theorem mapRange.addEquiv_toEquiv (f : M ≃+ N) :
    ↑(mapRange.addEquiv f : (α →₀ _) ≃+ _) =
      (mapRange.equiv (f : M ≃ N) f.map_zero f.symm.map_zero : (α →₀ _) ≃ _) :=
  Equiv.ext fun _ => rfl


/-- Given `f : α ≃ β`, we can map `l : α →₀ M` to `equivMapDomain f l : β →₀ M` (computably)
by mapping the support forwards and the function backwards. -/
def equivMapDomain (f : α ≃ β) (l : α →₀ M) : β →₀ M where
  support := l.support.map f.toEmbedding
  toFun a := l (f.symm a)
                            /-
                              α : Type u_1
                              β : Type u_2
                              γ : Type u_3
                              ι : Type u_4
                              M : Type u_5
                              M' : Type u_6
                              N : Type u_7
                              P : Type u_8
                              G : Type u_9
                              H : Type u_10
                              R : Type u_11
                              S : Type u_12
                              inst✝ : Zero M
                              f : Equiv α β
                              l : Finsupp α M
                              a : β
                              ⊢ Iff (Membership.mem (Finset.map f.toEmbedding l.support) a) (Ne ((fun a => l …
                            -/
  mem_support_toFun a := by simp only [Finset.mem_map_equiv, mem_support_toFun]; rfl
                                                                                 /-
                                                                                   🎉 no goals
                                                                                 -/


@[simp]
theorem equivMapDomain_apply (f : α ≃ β) (l : α →₀ M) (b : β) :
    equivMapDomain f l b = l (f.symm b) :=
  rfl


theorem equivMapDomain_symm_apply (f : α ≃ β) (l : β →₀ M) (a : α) :
    equivMapDomain f.symm l a = l (f a) :=
  rfl


@[simp]
                                                                                     /-
                                                                                       α : Type u_1
                                                                                       M : Type u_5
                                                                                       inst✝ : Zero M
                                                                                       l : Finsupp α M
                                                                                       ⊢ Eq (Finsupp.equivMapDomain (Equiv.refl α) l) l
                                                                                     -/
theorem equivMapDomain_refl (l : α →₀ M) : equivMapDomain (Equiv.refl _) l = l := by ext x; rfl
                                                                                            /-
                                                                                              🎉 no goals
                                                                                            -/


                                                                                  /-
                                                                                    α : Type u_1
                                                                                    M : Type u_5
                                                                                    inst✝ : Zero M
                                                                                    ⊢ Eq (Finsupp.equivMapDomain (Equiv.refl α)) id
                                                                                  -/
theorem equivMapDomain_refl' : equivMapDomain (Equiv.refl _) = @id (α →₀ M) := by ext x; rfl
                                                                                         /-
                                                                                           🎉 no goals
                                                                                         -/


theorem equivMapDomain_trans (f : α ≃ β) (g : β ≃ γ) (l : α →₀ M) :
                                                                               /-
                                                                                 α : Type u_1
                                                                                 β : Type u_2
                                                                                 γ : Type u_3
                                                                                 M : Type u_5
                                                                                 inst✝ : Zero M
                                                                                 f : Equiv α β
                                                                                 g : Equiv β γ
                                                                                 l : Finsupp α M
                                                                                 ⊢ Eq (Finsupp.equivMapDomain (f.trans g) l) (Finsupp.equivMapDomain g (Finsupp …
                                                                               -/
    equivMapDomain (f.trans g) l = equivMapDomain g (equivMapDomain f l) := by ext x; rfl
                                                                                      /-
                                                                                        🎉 no goals
                                                                                      -/


theorem equivMapDomain_trans' (f : α ≃ β) (g : β ≃ γ) :
                                                                                    /-
                                                                                      α : Type u_1
                                                                                      β : Type u_2
                                                                                      γ : Type u_3
                                                                                      M : Type u_5
                                                                                      inst✝ : Zero M
                                                                                      f : Equiv α β
                                                                                      g : Equiv β γ
                                                                                      ⊢ Eq (Finsupp.equivMapDomain (f.trans g)) (Function.comp (Finsupp.equivMapDoma …
                                                                                    -/
    @equivMapDomain _ _ M _ (f.trans g) = equivMapDomain g ∘ equivMapDomain f := by ext x; rfl
                                                                                           /-
                                                                                             🎉 no goals
                                                                                           -/


@[simp]
theorem equivMapDomain_single (f : α ≃ β) (a : α) (b : M) :
    equivMapDomain f (single a b) = single (f a) b := by
  classical
    ext x
    simp only [single_apply, Equiv.apply_eq_iff_eq_symm_apply, equivMapDomain_apply]


@[simp]
theorem equivMapDomain_zero {f : α ≃ β} : equivMapDomain f (0 : α →₀ M) = (0 : β →₀ M) := by
  /-
    α : Type u_1
    β : Type u_2
    M : Type u_5
    inst✝ : Zero M
    f : Equiv α β
    ⊢ Eq (Finsupp.equivMapDomain f 0) 0
  -/
  ext; simp only [equivMapDomain_apply, coe_zero, Pi.zero_apply]
       /-
         🎉 no goals
       -/


@[to_additive (attr := simp)]
theorem prod_equivMapDomain [CommMonoid N] (f : α ≃ β) (l : α →₀ M) (g : β → M → N) :
    prod (equivMapDomain f l) g = prod l (fun a m => g (f a) m) := by
  /-
    α : Type u_1
    β : Type u_2
    M : Type u_5
    N : Type u_7
    inst✝¹ : Zero M
    inst✝ : CommMonoid N
    f : Equiv α β
    l : Finsupp α M
    g : β → M → N
    ⊢ Eq ((Finsupp.equivMapDomain f l).prod g) (l.prod fun a m => g (f a) m)
  -/
  simp [prod, equivMapDomain]
  /-
    🎉 no goals
  -/


/-- Given `f : α ≃ β`, the finitely supported function spaces are also in bijection:
`(α →₀ M) ≃ (β →₀ M)`.

This is the finitely-supported version of `Equiv.piCongrLeft`. -/
def equivCongrLeft (f : α ≃ β) : (α →₀ M) ≃ (β →₀ M) := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    ι : Type u_4
    M : Type u_5
    M' : Type u_6
    N : Type u_7
    P : Type u_8
    G : Type u_9
    H : Type u_10
    R : Type u_11
    S : Type u_12
    inst✝ : Zero M
    f : Equiv α β
    ⊢ Equiv (Finsupp α M) (Finsupp β M)
  -/
  refine ⟨equivMapDomain f, equivMapDomain f.symm, fun f => ?_, fun f => ?_⟩ <;> ext x <;>
    simp only [equivMapDomain_apply, Equiv.symm_symm, Equiv.symm_apply_apply,
      Equiv.apply_symm_apply]


@[simp]
theorem equivCongrLeft_apply (f : α ≃ β) (l : α →₀ M) : equivCongrLeft f l = equivMapDomain f l :=
  rfl


@[simp]
theorem equivCongrLeft_symm (f : α ≃ β) :
    (@equivCongrLeft _ _ M _ f).symm = equivCongrLeft f.symm :=
  rfl


@[simp, norm_cast]
theorem cast_finsupp_prod [CommSemiring R] (g : α → M → ℕ) :
    (↑(f.prod g) : R) = f.prod fun a b => ↑(g a b) :=
  Nat.cast_prod _ _


@[simp, norm_cast]
theorem cast_finsupp_sum [CommSemiring R] (g : α → M → ℕ) :
    (↑(f.sum g) : R) = f.sum fun a b => ↑(g a b) :=
  Nat.cast_sum _ _


@[simp, norm_cast]
theorem cast_finsupp_prod [CommRing R] (g : α → M → ℤ) :
    (↑(f.prod g) : R) = f.prod fun a b => ↑(g a b) :=
  Int.cast_prod _ _


@[simp, norm_cast]
theorem cast_finsupp_sum [CommRing R] (g : α → M → ℤ) :
    (↑(f.sum g) : R) = f.sum fun a b => ↑(g a b) :=
  Int.cast_sum _ _


@[simp, norm_cast]
theorem cast_finsupp_sum [DivisionRing R] [CharZero R] (g : α → M → ℚ) :
    (↑(f.sum g) : R) = f.sum fun a b => ↑(g a b) :=
  cast_sum _ _


@[simp, norm_cast]
theorem cast_finsupp_prod [Field R] [CharZero R] (g : α → M → ℚ) :
    (↑(f.prod g) : R) = f.prod fun a b => ↑(g a b) :=
  cast_prod _ _


/-- Given `f : α → β` and `v : α →₀ M`, `mapDomain f v : β →₀ M`
  is the finitely supported function whose value at `a : β` is the sum
  of `v x` over all `x` such that `f x = a`. -/
def mapDomain (f : α → β) (v : α →₀ M) : β →₀ M :=
  v.sum fun a => single (f a)


theorem mapDomain_apply {f : α → β} (hf : Function.Injective f) (x : α →₀ M) (a : α) :
    mapDomain f x (f a) = x a := by
  /-
    α : Type u_1
    β : Type u_2
    M : Type u_5
    inst✝ : AddCommMonoid M
    f : α → β
    hf : Function.Injective f
    x : Finsupp α M
    a : α
    ⊢ Eq ((Finsupp.mapDomain f x) (f a)) (x a)
  -/
  rw [mapDomain, sum_apply, sum_eq_single a, single_eq_same]
    /-
      case h₀
      α : Type u_1
      β : Type u_2
      M : Type u_5
      inst✝ : AddCommMonoid M
      f : α → β
      hf : Function.Injective f
      x : Finsupp α M
      a : α
      ⊢ ∀ (b : α), Ne (x b) 0 → Ne b a → Eq ((Finsupp.single (f b) (x b)) (f a)) 0
    -/
  · intro b _ hba
    /-
      case h₀
      α : Type u_1
      β : Type u_2
      M : Type u_5
      inst✝ : AddCommMonoid M
      f : α → β
      hf : Function.Injective f
      x : Finsupp α M
      a b : α
      a✝ : Ne (x b) 0
      hba : Ne b a
      ⊢ Eq ((Finsupp.single (f b) (x b)) (f a)) 0
    -/
    exact single_eq_of_ne (hf.ne hba)
    /-
      🎉 no goals
    -/
    /-
      case h₁
      α : Type u_1
      β : Type u_2
      M : Type u_5
      inst✝ : AddCommMonoid M
      f : α → β
      hf : Function.Injective f
      x : Finsupp α M
      a : α
      ⊢ Eq (x a) 0 → Eq ((Finsupp.single (f a) 0) (f a)) 0
    -/
  · intro _
    /-
      case h₁
      α : Type u_1
      β : Type u_2
      M : Type u_5
      inst✝ : AddCommMonoid M
      f : α → β
      hf : Function.Injective f
      x : Finsupp α M
      a : α
      a✝ : Eq (x a) 0
      ⊢ Eq ((Finsupp.single (f a) 0) (f a)) 0
    -/
    rw [single_zero, coe_zero, Pi.zero_apply]
    /-
      🎉 no goals
    -/


theorem mapDomain_notin_range {f : α → β} (x : α →₀ M) (a : β) (h : a ∉ Set.range f) :
    mapDomain f x a = 0 := by
  /-
    α : Type u_1
    β : Type u_2
    M : Type u_5
    inst✝ : AddCommMonoid M
    f : α → β
    x : Finsupp α M
    a : β
    h : Not (Membership.mem (Set.range f) a)
    ⊢ Eq ((Finsupp.mapDomain f x) a) 0
  -/
  rw [mapDomain, sum_apply, sum]
  /-
    α : Type u_1
    β : Type u_2
    M : Type u_5
    inst✝ : AddCommMonoid M
    f : α → β
    x : Finsupp α M
    a : β
    h : Not (Membership.mem (Set.range f) a)
    ⊢ Eq (x.support.sum fun a_1 => (Finsupp.single (f a_1) (x a_1)) a) 0
  -/
  exact Finset.sum_eq_zero fun a' _ => single_eq_of_ne fun eq => h <| eq ▸ Set.mem_range_self _
  /-
    🎉 no goals
  -/


@[simp]
theorem mapDomain_id : mapDomain id v = v :=
  sum_single _


theorem mapDomain_comp {f : α → β} {g : β → γ} :
    mapDomain (g ∘ f) v = mapDomain g (mapDomain f v) := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    M : Type u_5
    inst✝ : AddCommMonoid M
    v : Finsupp α M
    f : α → β
    g : β → γ
    ⊢ Eq (Finsupp.mapDomain (Function.comp g f) v) (Finsupp.mapDomain g (Finsupp.m …
  -/
  refine ((sum_sum_index ?_ ?_).trans ?_).symm
    /-
      case refine_1
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      M : Type u_5
      inst✝ : AddCommMonoid M
      v : Finsupp α M
      f : α → β
      g : β → γ
      ⊢ ∀ (a : β), Eq (Finsupp.single (g a) 0) 0
    -/
  · intro
    /-
      case refine_1
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      M : Type u_5
      inst✝ : AddCommMonoid M
      v : Finsupp α M
      f : α → β
      g : β → γ
      a✝ : β
      ⊢ Eq (Finsupp.single (g a✝) 0) 0
    -/
    exact single_zero _
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      M : Type u_5
      inst✝ : AddCommMonoid M
      v : Finsupp α M
      f : α → β
      g : β → γ
      ⊢ ∀ (a : β) (b₁ b₂ : M), Eq (Finsupp.single (g a) (HAdd.hAdd b₁ b₂)) (HAdd.hAd …
    -/
  · intro
    /-
      case refine_2
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      M : Type u_5
      inst✝ : AddCommMonoid M
      v : Finsupp α M
      f : α → β
      g : β → γ
      a✝ : β
      ⊢ ∀ (b₁ b₂ : M), Eq (Finsupp.single (g a✝) (HAdd.hAdd b₁ b₂)) (HAdd.hAdd (Fins …
    -/
    exact single_add _
    /-
      🎉 no goals
    -/
  /-
    case refine_3
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    M : Type u_5
    inst✝ : AddCommMonoid M
    v : Finsupp α M
    f : α → β
    g : β → γ
    ⊢ Eq (v.sum fun a b => (Finsupp.single (f a) b).sum fun a => Finsupp.single (g …
  -/
  refine sum_congr fun _ _ => sum_single_index ?_
  /-
    case refine_3
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    M : Type u_5
    inst✝ : AddCommMonoid M
    v : Finsupp α M
    f : α → β
    g : β → γ
    x✝¹ : α
    x✝ : Membership.mem v.support x✝¹
    ⊢ Eq (Finsupp.single (g (f x✝¹)) 0) 0
  -/
  exact single_zero _
  /-
    🎉 no goals
  -/


@[simp]
theorem mapDomain_single {f : α → β} {a : α} {b : M} : mapDomain f (single a b) = single (f a) b :=
  sum_single_index <| single_zero _


@[simp]
theorem mapDomain_zero {f : α → β} : mapDomain f (0 : α →₀ M) = (0 : β →₀ M) :=
  sum_zero_index


theorem mapDomain_congr {f g : α → β} (h : ∀ x ∈ v.support, f x = g x) :
    v.mapDomain f = v.mapDomain g :=
                                     /-
                                       α : Type u_1
                                       β : Type u_2
                                       M : Type u_5
                                       inst✝ : AddCommMonoid M
                                       v : Finsupp α M
                                       f g : α → β
                                       h : ∀ (x : α), Membership.mem v.support x → Eq (f x) (g x)
                                       x✝ : α
                                       H : Membership.mem v.support x✝
                                       ⊢ Eq ((fun a => Finsupp.single (f a)) x✝ (v x✝)) ((fun a => Finsupp.single (g  …
                                     -/
  Finset.sum_congr rfl fun _ H => by simp only [h _ H]
                                     /-
                                       🎉 no goals
                                     -/


theorem mapDomain_add {f : α → β} : mapDomain f (v₁ + v₂) = mapDomain f v₁ + mapDomain f v₂ :=
  sum_add_index' (fun _ => single_zero _) fun _ => single_add _


@[simp]
theorem mapDomain_equiv_apply {f : α ≃ β} (x : α →₀ M) (a : β) :
    mapDomain f x a = x (f.symm a) := by
  /-
    α : Type u_1
    β : Type u_2
    M : Type u_5
    inst✝ : AddCommMonoid M
    f : Equiv α β
    x : Finsupp α M
    a : β
    ⊢ Eq ((Finsupp.mapDomain (⇑f) x) a) (x (f.symm a))
  -/
  conv_lhs => rw [← f.apply_symm_apply a]
  /-
    α : Type u_1
    β : Type u_2
    M : Type u_5
    inst✝ : AddCommMonoid M
    f : Equiv α β
    x : Finsupp α M
    a : β
    ⊢ Eq ((Finsupp.mapDomain (⇑f) x) (f (f.symm a))) (x (f.symm a))
  -/
  exact mapDomain_apply f.injective _ _
  /-
    🎉 no goals
  -/


/-- `Finsupp.mapDomain` is an `AddMonoidHom`. -/
@[simps]
def mapDomain.addMonoidHom (f : α → β) : (α →₀ M) →+ β →₀ M where
  toFun := mapDomain f
  map_zero' := mapDomain_zero
  map_add' _ _ := mapDomain_add


@[simp]
theorem mapDomain.addMonoidHom_id : mapDomain.addMonoidHom id = AddMonoidHom.id (α →₀ M) :=
  AddMonoidHom.ext fun _ => mapDomain_id


theorem mapDomain.addMonoidHom_comp (f : β → γ) (g : α → β) :
    (mapDomain.addMonoidHom (f ∘ g) : (α →₀ M) →+ γ →₀ M) =
      (mapDomain.addMonoidHom f).comp (mapDomain.addMonoidHom g) :=
  AddMonoidHom.ext fun _ => mapDomain_comp


theorem mapDomain_finset_sum {f : α → β} {s : Finset ι} {v : ι → α →₀ M} :
    mapDomain f (∑ i ∈ s, v i) = ∑ i ∈ s, mapDomain f (v i) :=
  map_sum (mapDomain.addMonoidHom f) _ _


theorem mapDomain_sum [Zero N] {f : α → β} {s : α →₀ N} {v : α → N → α →₀ M} :
    mapDomain f (s.sum v) = s.sum fun a b => mapDomain f (v a b) :=
  map_finsupp_sum (mapDomain.addMonoidHom f : (α →₀ M) →+ β →₀ M) _ _


theorem mapDomain_support [DecidableEq β] {f : α → β} {s : α →₀ M} :
    (s.mapDomain f).support ⊆ s.support.image f :=
  Finset.Subset.trans support_sum <|
    Finset.Subset.trans (Finset.biUnion_mono fun _ _ => support_single_subset) <| by
      /-
        α : Type u_1
        β : Type u_2
        M : Type u_5
        inst✝¹ : AddCommMonoid M
        inst✝ : DecidableEq β
        f : α → β
        s : Finsupp α M
        ⊢ HasSubset.Subset (s.support.biUnion fun x => Singleton.singleton (f x)) (Fin …
      -/
      rw [Finset.biUnion_singleton]
      /-
        🎉 no goals
      -/


theorem mapDomain_apply' (S : Set α) {f : α → β} (x : α →₀ M) (hS : (x.support : Set α) ⊆ S)
    (hf : Set.InjOn f S) {a : α} (ha : a ∈ S) : mapDomain f x (f a) = x a := by
  classical
    rw [mapDomain, sum_apply, sum]
    simp_rw [single_apply]
    by_cases hax : a ∈ x.support
    · rw [← Finset.add_sum_erase _ _ hax, if_pos rfl]
      convert add_zero (x a)
      refine Finset.sum_eq_zero fun i hi => if_neg ?_
      exact (hf.mono hS).ne (Finset.mem_of_mem_erase hi) hax (Finset.ne_of_mem_erase hi)
    · rw [not_mem_support_iff.1 hax]
      refine Finset.sum_eq_zero fun i hi => if_neg ?_
      exact hf.ne (hS hi) ha (ne_of_mem_of_not_mem hi hax)


theorem mapDomain_support_of_injOn [DecidableEq β] {f : α → β} (s : α →₀ M)
    (hf : Set.InjOn f s.support) : (mapDomain f s).support = Finset.image f s.support :=
  Finset.Subset.antisymm mapDomain_support <| by
    /-
      α : Type u_1
      β : Type u_2
      M : Type u_5
      inst✝¹ : AddCommMonoid M
      inst✝ : DecidableEq β
      f : α → β
      s : Finsupp α M
      hf : Set.InjOn f ↑s.support
      ⊢ HasSubset.Subset (Finset.image f s.support) (Finsupp.mapDomain f s).support
    -/
    intro x hx
    /-
      α : Type u_1
      β : Type u_2
      M : Type u_5
      inst✝¹ : AddCommMonoid M
      inst✝ : DecidableEq β
      f : α → β
      s : Finsupp α M
      hf : Set.InjOn f ↑s.support
      x : β
      hx : Membership.mem (Finset.image f s.support) x
      ⊢ Membership.mem (Finsupp.mapDomain f s).support x
    -/
    simp only [mem_image, exists_prop, mem_support_iff, Ne] at hx
    /-
      α : Type u_1
      β : Type u_2
      M : Type u_5
      inst✝¹ : AddCommMonoid M
      inst✝ : DecidableEq β
      f : α → β
      s : Finsupp α M
      hf : Set.InjOn f ↑s.support
      x : β
      hx : Exists fun a => And (Not (Eq (s a) 0)) (Eq (f a) x)
      ⊢ Membership.mem (Finsupp.mapDomain f s).support x
    -/
    rcases hx with ⟨hx_w, hx_h_left, rfl⟩
    /-
      case intro.intro
      α : Type u_1
      β : Type u_2
      M : Type u_5
      inst✝¹ : AddCommMonoid M
      inst✝ : DecidableEq β
      f : α → β
      s : Finsupp α M
      hf : Set.InjOn f ↑s.support
      hx_w : α
      hx_h_left : Not (Eq (s hx_w) 0)
      ⊢ Membership.mem (Finsupp.mapDomain f s).support (f hx_w)
    -/
    simp only [mem_support_iff, Ne]
    /-
      case intro.intro
      α : Type u_1
      β : Type u_2
      M : Type u_5
      inst✝¹ : AddCommMonoid M
      inst✝ : DecidableEq β
      f : α → β
      s : Finsupp α M
      hf : Set.InjOn f ↑s.support
      hx_w : α
      hx_h_left : Not (Eq (s hx_w) 0)
      ⊢ Not (Eq ((Finsupp.mapDomain f s) (f hx_w)) 0)
    -/
    rw [mapDomain_apply' (↑s.support : Set _) _ _ hf]
      /-
        case intro.intro
        α : Type u_1
        β : Type u_2
        M : Type u_5
        inst✝¹ : AddCommMonoid M
        inst✝ : DecidableEq β
        f : α → β
        s : Finsupp α M
        hf : Set.InjOn f ↑s.support
        hx_w : α
        hx_h_left : Not (Eq (s hx_w) 0)
        ⊢ Not (Eq (s hx_w) 0)
      -/
    · exact hx_h_left
      /-
        🎉 no goals
      -/
      /-
        case intro.intro
        α : Type u_1
        β : Type u_2
        M : Type u_5
        inst✝¹ : AddCommMonoid M
        inst✝ : DecidableEq β
        f : α → β
        s : Finsupp α M
        hf : Set.InjOn f ↑s.support
        hx_w : α
        hx_h_left : Not (Eq (s hx_w) 0)
        ⊢ Membership.mem (↑s.support) hx_w
      -/
    · simp only [mem_coe, mem_support_iff, Ne]
      /-
        case intro.intro
        α : Type u_1
        β : Type u_2
        M : Type u_5
        inst✝¹ : AddCommMonoid M
        inst✝ : DecidableEq β
        f : α → β
        s : Finsupp α M
        hf : Set.InjOn f ↑s.support
        hx_w : α
        hx_h_left : Not (Eq (s hx_w) 0)
        ⊢ Not (Eq (s hx_w) 0)
      -/
      exact hx_h_left
      /-
        🎉 no goals
      -/
      /-
        α : Type u_1
        β : Type u_2
        M : Type u_5
        inst✝¹ : AddCommMonoid M
        inst✝ : DecidableEq β
        f : α → β
        s : Finsupp α M
        hf : Set.InjOn f ↑s.support
        hx_w : α
        hx_h_left : Not (Eq (s hx_w) 0)
        ⊢ HasSubset.Subset ↑s.support ↑s.support
      -/
    · exact Subset.refl _
      /-
        🎉 no goals
      -/


theorem mapDomain_support_of_injective [DecidableEq β] {f : α → β} (hf : Function.Injective f)
    (s : α →₀ M) : (mapDomain f s).support = Finset.image f s.support :=
  mapDomain_support_of_injOn s hf.injOn


@[to_additive]
theorem prod_mapDomain_index [CommMonoid N] {f : α → β} {s : α →₀ M} {h : β → M → N}
    (h_zero : ∀ b, h b 0 = 1) (h_add : ∀ b m₁ m₂, h b (m₁ + m₂) = h b m₁ * h b m₂) :
    (mapDomain f s).prod h = s.prod fun a m => h (f a) m :=
  (prod_sum_index h_zero h_add).trans <| prod_congr fun _ _ => prod_single_index (h_zero _)

-- Note that in `prod_mapDomain_index`, `M` is still an additive monoid,
-- so there is no analogous version in terms of `MonoidHom`.

/-- A version of `sum_mapDomain_index` that takes a bundled `AddMonoidHom`,
rather than separate linearity hypotheses.
-/
@[simp]
theorem sum_mapDomain_index_addMonoidHom [AddCommMonoid N] {f : α → β} {s : α →₀ M}
    (h : β → M →+ N) : ((mapDomain f s).sum fun b m => h b m) = s.sum fun a m => h (f a) m :=
  sum_mapDomain_index (fun b => (h b).map_zero) (fun b _ _ => (h b).map_add _ _)


theorem embDomain_eq_mapDomain (f : α ↪ β) (v : α →₀ M) : embDomain f v = mapDomain f v := by
  /-
    α : Type u_1
    β : Type u_2
    M : Type u_5
    inst✝ : AddCommMonoid M
    f : Function.Embedding α β
    v : Finsupp α M
    ⊢ Eq (Finsupp.embDomain f v) (Finsupp.mapDomain (⇑f) v)
  -/
  ext a
  /-
    case h
    α : Type u_1
    β : Type u_2
    M : Type u_5
    inst✝ : AddCommMonoid M
    f : Function.Embedding α β
    v : Finsupp α M
    a : β
    ⊢ Eq ((Finsupp.embDomain f v) a) ((Finsupp.mapDomain (⇑f) v) a)
  -/
  by_cases h : a ∈ Set.range f
    /-
      case pos
      α : Type u_1
      β : Type u_2
      M : Type u_5
      inst✝ : AddCommMonoid M
      f : Function.Embedding α β
      v : Finsupp α M
      a : β
      h : Membership.mem (Set.range ⇑f) a
      ⊢ Eq ((Finsupp.embDomain f v) a) ((Finsupp.mapDomain (⇑f) v) a)
    -/
  · rcases h with ⟨a, rfl⟩
    /-
      case pos.intro
      α : Type u_1
      β : Type u_2
      M : Type u_5
      inst✝ : AddCommMonoid M
      f : Function.Embedding α β
      v : Finsupp α M
      a : α
      ⊢ Eq ((Finsupp.embDomain f v) (f a)) ((Finsupp.mapDomain (⇑f) v) (f a))
    -/
    rw [mapDomain_apply f.injective, embDomain_apply]
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      β : Type u_2
      M : Type u_5
      inst✝ : AddCommMonoid M
      f : Function.Embedding α β
      v : Finsupp α M
      a : β
      h : Not (Membership.mem (Set.range ⇑f) a)
      ⊢ Eq ((Finsupp.embDomain f v) a) ((Finsupp.mapDomain (⇑f) v) a)
    -/
                                                          /-
                                                            🎉 no goals
                                                          -/
  · rw [mapDomain_notin_range, embDomain_notin_range] <;> assumption
                                                          /-
                                                            🎉 no goals
                                                          -/


@[to_additive]
theorem prod_mapDomain_index_inj [CommMonoid N] {f : α → β} {s : α →₀ M} {h : β → M → N}
    (hf : Function.Injective f) : (s.mapDomain f).prod h = s.prod fun a b => h (f a) b := by
  /-
    α : Type u_1
    β : Type u_2
    M : Type u_5
    N : Type u_7
    inst✝¹ : AddCommMonoid M
    inst✝ : CommMonoid N
    f : α → β
    s : Finsupp α M
    h : β → M → N
    hf : Function.Injective f
    ⊢ Eq ((Finsupp.mapDomain f s).prod h) (s.prod fun a b => h (f a) b)
  -/
  rw [← Function.Embedding.coeFn_mk f hf, ← embDomain_eq_mapDomain, prod_embDomain]
  /-
    🎉 no goals
  -/


theorem mapDomain_injective {f : α → β} (hf : Function.Injective f) :
    Function.Injective (mapDomain f : (α →₀ M) → β →₀ M) := by
  /-
    α : Type u_1
    β : Type u_2
    M : Type u_5
    inst✝ : AddCommMonoid M
    f : α → β
    hf : Function.Injective f
    ⊢ Function.Injective (Finsupp.mapDomain f)
  -/
  intro v₁ v₂ eq
  /-
    α : Type u_1
    β : Type u_2
    M : Type u_5
    inst✝ : AddCommMonoid M
    f : α → β
    hf : Function.Injective f
    v₁ v₂ : Finsupp α M
    eq : Eq (Finsupp.mapDomain f v₁) (Finsupp.mapDomain f v₂)
    ⊢ Eq v₁ v₂
  -/
  ext a
  /-
    case h
    α : Type u_1
    β : Type u_2
    M : Type u_5
    inst✝ : AddCommMonoid M
    f : α → β
    hf : Function.Injective f
    v₁ v₂ : Finsupp α M
    eq : Eq (Finsupp.mapDomain f v₁) (Finsupp.mapDomain f v₂)
    a : α
    ⊢ Eq (v₁ a) (v₂ a)
  -/
  have : mapDomain f v₁ (f a) = mapDomain f v₂ (f a) := by rw [eq]
  /-
    case h
    α : Type u_1
    β : Type u_2
    M : Type u_5
    inst✝ : AddCommMonoid M
    f : α → β
    hf : Function.Injective f
    v₁ v₂ : Finsupp α M
    eq : Eq (Finsupp.mapDomain f v₁) (Finsupp.mapDomain f v₂)
    a : α
    this : Eq ((Finsupp.mapDomain f v₁) (f a)) ((Finsupp.mapDomain f v₂) (f a))
    ⊢ Eq (v₁ a) (v₂ a)
  -/
  rwa [mapDomain_apply hf, mapDomain_apply hf] at this
  /-
    🎉 no goals
  -/


/-- When `f` is an embedding we have an embedding `(α →₀ ℕ) ↪ (β →₀ ℕ)` given by `mapDomain`. -/
@[simps]
def mapDomainEmbedding {α β : Type*} (f : α ↪ β) : (α →₀ ℕ) ↪ β →₀ ℕ :=
  ⟨Finsupp.mapDomain f, Finsupp.mapDomain_injective f.injective⟩


theorem mapDomain.addMonoidHom_comp_mapRange [AddCommMonoid N] (f : α → β) (g : M →+ N) :
    (mapDomain.addMonoidHom f).comp (mapRange.addMonoidHom g) =
      (mapRange.addMonoidHom g).comp (mapDomain.addMonoidHom f) := by
  /-
    α : Type u_1
    β : Type u_2
    M : Type u_5
    N : Type u_7
    inst✝¹ : AddCommMonoid M
    inst✝ : AddCommMonoid N
    f : α → β
    g : AddMonoidHom M N
    ⊢ Eq ((Finsupp.mapDomain.addMonoidHom f).comp (Finsupp.mapRange.addMonoidHom g …
  -/
  ext
  simp only [AddMonoidHom.coe_comp, Finsupp.mapRange_single, Finsupp.mapDomain.addMonoidHom_apply,
    Finsupp.singleAddHom_apply, eq_self_iff_true, Function.comp_apply, Finsupp.mapDomain_single,
    Finsupp.mapRange.addMonoidHom_apply]


/-- When `g` preserves addition, `mapRange` and `mapDomain` commute. -/
theorem mapDomain_mapRange [AddCommMonoid N] (f : α → β) (v : α →₀ M) (g : M → N) (h0 : g 0 = 0)
    (hadd : ∀ x y, g (x + y) = g x + g y) :
    mapDomain f (mapRange g h0 v) = mapRange g h0 (mapDomain f v) :=
  let g' : M →+ N :=
    { toFun := g
      map_zero' := h0
      map_add' := hadd }
  DFunLike.congr_fun (mapDomain.addMonoidHom_comp_mapRange f g') v


theorem sum_update_add [AddCommMonoid α] [AddCommMonoid β] (f : ι →₀ α) (i : ι) (a : α)
    (g : ι → α → β) (hg : ∀ i, g i 0 = 0)
    (hgg : ∀ (j : ι) (a₁ a₂ : α), g j (a₁ + a₂) = g j a₁ + g j a₂) :
    (f.update i a).sum g + g i (f i) = f.sum g + g i a := by
  /-
    α : Type u_1
    β : Type u_2
    ι : Type u_4
    inst✝¹ : AddCommMonoid α
    inst✝ : AddCommMonoid β
    f : Finsupp ι α
    i : ι
    a : α
    g : ι → α → β
    hg : ∀ (i : ι), Eq (g i 0) 0
    hgg : ∀ (j : ι) (a₁ a₂ : α), Eq (g j (HAdd.hAdd a₁ a₂)) (HAdd.hAdd (g j a₁) (g …
    ⊢ Eq (HAdd.hAdd ((f.update i a).sum g) (g i (f i))) (HAdd.hAdd (f.sum g) (g i  …
  -/
  rw [update_eq_erase_add_single, sum_add_index' hg hgg]
  /-
    α : Type u_1
    β : Type u_2
    ι : Type u_4
    inst✝¹ : AddCommMonoid α
    inst✝ : AddCommMonoid β
    f : Finsupp ι α
    i : ι
    a : α
    g : ι → α → β
    hg : ∀ (i : ι), Eq (g i 0) 0
    hgg : ∀ (j : ι) (a₁ a₂ : α), Eq (g j (HAdd.hAdd a₁ a₂)) (HAdd.hAdd (g j a₁) (g …
    ⊢ Eq (HAdd.hAdd (HAdd.hAdd ((Finsupp.erase i f).sum g) ((Finsupp.single i a).s …
  -/
  conv_rhs => rw [← Finsupp.update_self f i]
  /-
    α : Type u_1
    β : Type u_2
    ι : Type u_4
    inst✝¹ : AddCommMonoid α
    inst✝ : AddCommMonoid β
    f : Finsupp ι α
    i : ι
    a : α
    g : ι → α → β
    hg : ∀ (i : ι), Eq (g i 0) 0
    hgg : ∀ (j : ι) (a₁ a₂ : α), Eq (g j (HAdd.hAdd a₁ a₂)) (HAdd.hAdd (g j a₁) (g …
    ⊢ Eq (HAdd.hAdd (HAdd.hAdd ((Finsupp.erase i f).sum g) ((Finsupp.single i a).s …
  -/
  rw [update_eq_erase_add_single, sum_add_index' hg hgg, add_assoc, add_assoc]
  /-
    α : Type u_1
    β : Type u_2
    ι : Type u_4
    inst✝¹ : AddCommMonoid α
    inst✝ : AddCommMonoid β
    f : Finsupp ι α
    i : ι
    a : α
    g : ι → α → β
    hg : ∀ (i : ι), Eq (g i 0) 0
    hgg : ∀ (j : ι) (a₁ a₂ : α), Eq (g j (HAdd.hAdd a₁ a₂)) (HAdd.hAdd (g j a₁) (g …
    ⊢ Eq (HAdd.hAdd ((Finsupp.erase i f).sum g) (HAdd.hAdd ((Finsupp.single i a).s …
  -/
  congr 1
  /-
    case e_a
    α : Type u_1
    β : Type u_2
    ι : Type u_4
    inst✝¹ : AddCommMonoid α
    inst✝ : AddCommMonoid β
    f : Finsupp ι α
    i : ι
    a : α
    g : ι → α → β
    hg : ∀ (i : ι), Eq (g i 0) 0
    hgg : ∀ (j : ι) (a₁ a₂ : α), Eq (g j (HAdd.hAdd a₁ a₂)) (HAdd.hAdd (g j a₁) (g …
    ⊢ Eq (HAdd.hAdd ((Finsupp.single i a).sum g) (g i (f i))) (HAdd.hAdd ((Finsupp …
  -/
  rw [add_comm, sum_single_index (hg _), sum_single_index (hg _)]
  /-
    🎉 no goals
  -/


theorem mapDomain_injOn (S : Set α) {f : α → β} (hf : Set.InjOn f S) :
    Set.InjOn (mapDomain f : (α →₀ M) → β →₀ M) { w | (w.support : Set α) ⊆ S } := by
  /-
    α : Type u_1
    β : Type u_2
    M : Type u_5
    inst✝ : AddCommMonoid M
    S : Set α
    f : α → β
    hf : Set.InjOn f S
    ⊢ Set.InjOn (Finsupp.mapDomain f) (setOf fun w => HasSubset.Subset (↑w.support …
  -/
  intro v₁ hv₁ v₂ hv₂ eq
  /-
    α : Type u_1
    β : Type u_2
    M : Type u_5
    inst✝ : AddCommMonoid M
    S : Set α
    f : α → β
    hf : Set.InjOn f S
    v₁ : Finsupp α M
    hv₁ : Membership.mem (setOf fun w => HasSubset.Subset (↑w.support) S) v₁
    v₂ : Finsupp α M
    hv₂ : Membership.mem (setOf fun w => HasSubset.Subset (↑w.support) S) v₂
    eq : Eq (Finsupp.mapDomain f v₁) (Finsupp.mapDomain f v₂)
    ⊢ Eq v₁ v₂
  -/
  ext a
  classical
    by_cases h : a ∈ v₁.support ∪ v₂.support
    · rw [← mapDomain_apply' S _ hv₁ hf _, ← mapDomain_apply' S _ hv₂ hf _, eq] <;>
        · apply Set.union_subset hv₁ hv₂
          exact mod_cast h
    · simp only [not_or, mem_union, not_not, mem_support_iff] at h
      simp [h]


theorem equivMapDomain_eq_mapDomain {M} [AddCommMonoid M] (f : α ≃ β) (l : α →₀ M) :
                                             /-
                                               α : Type u_1
                                               β : Type u_2
                                               M : Type u_13
                                               inst✝ : AddCommMonoid M
                                               f : Equiv α β
                                               l : Finsupp α M
                                               ⊢ Eq (Finsupp.equivMapDomain f l) (Finsupp.mapDomain (⇑f) l)
                                             -/
    equivMapDomain f l = mapDomain f l := by ext x; simp [mapDomain_equiv_apply]
                                                    /-
                                                      🎉 no goals
                                                    -/


/-- Given `f : α → β`, `l : β →₀ M` and a proof `hf` that `f` is injective on
the preimage of `l.support`, `comapDomain f l hf` is the finitely supported function
from `α` to `M` given by composing `l` with `f`. -/
@[simps support]
def comapDomain [Zero M] (f : α → β) (l : β →₀ M) (hf : Set.InjOn f (f ⁻¹' ↑l.support)) :
    α →₀ M where
  support := l.support.preimage f hf
  toFun a := l (f a)
  mem_support_toFun := by
    /-
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      ι : Type u_4
      M : Type u_5
      M' : Type u_6
      N : Type u_7
      P : Type u_8
      G : Type u_9
      H : Type u_10
      R : Type u_11
      S : Type u_12
      inst✝ : Zero M
      f : α → β
      l : Finsupp β M
      hf : Set.InjOn f (Set.preimage f ↑l.support)
      ⊢ ∀ (a : α), Iff (Membership.mem (l.support.preimage f hf) a) (Ne ((fun a => l …
    -/
    intro a
    /-
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      ι : Type u_4
      M : Type u_5
      M' : Type u_6
      N : Type u_7
      P : Type u_8
      G : Type u_9
      H : Type u_10
      R : Type u_11
      S : Type u_12
      inst✝ : Zero M
      f : α → β
      l : Finsupp β M
      hf : Set.InjOn f (Set.preimage f ↑l.support)
      a : α
      ⊢ Iff (Membership.mem (l.support.preimage f hf) a) (Ne ((fun a => l (f a)) a) 0)
    -/
    simp only [Finset.mem_def.symm, Finset.mem_preimage]
    /-
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      ι : Type u_4
      M : Type u_5
      M' : Type u_6
      N : Type u_7
      P : Type u_8
      G : Type u_9
      H : Type u_10
      R : Type u_11
      S : Type u_12
      inst✝ : Zero M
      f : α → β
      l : Finsupp β M
      hf : Set.InjOn f (Set.preimage f ↑l.support)
      a : α
      ⊢ Iff (Membership.mem l.support (f a)) (Ne (l (f a)) 0)
    -/
    exact l.mem_support_toFun (f a)
    /-
      🎉 no goals
    -/


@[simp]
theorem comapDomain_apply [Zero M] (f : α → β) (l : β →₀ M) (hf : Set.InjOn f (f ⁻¹' ↑l.support))
    (a : α) : comapDomain f l hf a = l (f a) :=
  rfl


theorem sum_comapDomain [Zero M] [AddCommMonoid N] (f : α → β) (l : β →₀ M) (g : β → M → N)
    (hf : Set.BijOn f (f ⁻¹' ↑l.support) ↑l.support) :
    (comapDomain f l hf.injOn).sum (g ∘ f) = l.sum g := by
  /-
    α : Type u_1
    β : Type u_2
    M : Type u_5
    N : Type u_7
    inst✝¹ : Zero M
    inst✝ : AddCommMonoid N
    f : α → β
    l : Finsupp β M
    g : β → M → N
    hf : Set.BijOn f (Set.preimage f ↑l.support) ↑l.support
    ⊢ Eq ((Finsupp.comapDomain f l ⋯).sum (Function.comp g f)) (l.sum g)
  -/
  simp only [sum, comapDomain_apply, (· ∘ ·), comapDomain]
  /-
    α : Type u_1
    β : Type u_2
    M : Type u_5
    N : Type u_7
    inst✝¹ : Zero M
    inst✝ : AddCommMonoid N
    f : α → β
    l : Finsupp β M
    g : β → M → N
    hf : Set.BijOn f (Set.preimage f ↑l.support) ↑l.support
    ⊢ Eq ((l.support.preimage f ⋯).sum fun x => g (f x) ({ support := l.support.pr …
  -/
  exact Finset.sum_preimage_of_bij f _ hf fun x => g x (l x)
  /-
    🎉 no goals
  -/


theorem eq_zero_of_comapDomain_eq_zero [AddCommMonoid M] (f : α → β) (l : β →₀ M)
    (hf : Set.BijOn f (f ⁻¹' ↑l.support) ↑l.support) : comapDomain f l hf.injOn = 0 → l = 0 := by
  /-
    α : Type u_1
    β : Type u_2
    M : Type u_5
    inst✝ : AddCommMonoid M
    f : α → β
    l : Finsupp β M
    hf : Set.BijOn f (Set.preimage f ↑l.support) ↑l.support
    ⊢ Eq (Finsupp.comapDomain f l ⋯) 0 → Eq l 0
  -/
  rw [← support_eq_empty, ← support_eq_empty, comapDomain]
  /-
    α : Type u_1
    β : Type u_2
    M : Type u_5
    inst✝ : AddCommMonoid M
    f : α → β
    l : Finsupp β M
    hf : Set.BijOn f (Set.preimage f ↑l.support) ↑l.support
    ⊢ Eq { support := l.support.preimage f ⋯, toFun := fun a => l (f a), mem_suppo …
  -/
  simp only [Finset.ext_iff, Finset.not_mem_empty, iff_false, mem_preimage]
  /-
    α : Type u_1
    β : Type u_2
    M : Type u_5
    inst✝ : AddCommMonoid M
    f : α → β
    l : Finsupp β M
    hf : Set.BijOn f (Set.preimage f ↑l.support) ↑l.support
    ⊢ (∀ (a : α), Not (Membership.mem l.support (f a))) → ∀ (a : β), Not (Membersh …
  -/
  intro h a ha
  /-
    α : Type u_1
    β : Type u_2
    M : Type u_5
    inst✝ : AddCommMonoid M
    f : α → β
    l : Finsupp β M
    hf : Set.BijOn f (Set.preimage f ↑l.support) ↑l.support
    h : ∀ (a : α), Not (Membership.mem l.support (f a))
    a : β
    ha : Membership.mem l.support a
    ⊢ False
  -/
  cases' hf.2.2 ha with b hb
  /-
    case intro
    α : Type u_1
    β : Type u_2
    M : Type u_5
    inst✝ : AddCommMonoid M
    f : α → β
    l : Finsupp β M
    hf : Set.BijOn f (Set.preimage f ↑l.support) ↑l.support
    h : ∀ (a : α), Not (Membership.mem l.support (f a))
    a : β
    ha : Membership.mem l.support a
    b : α
    hb : And (Membership.mem (Set.preimage f ↑l.support) b) (Eq (f b) a)
    ⊢ False
  -/
  exact h b (hb.2.symm ▸ ha)
  /-
    🎉 no goals
  -/


lemma embDomain_comapDomain {f : α ↪ β} {g : β →₀ M} (hg : ↑g.support ⊆ Set.range f) :
    embDomain f (comapDomain f g f.injective.injOn) = g := by
  /-
    α : Type u_1
    β : Type u_2
    M : Type u_5
    inst✝ : Zero M
    f : Function.Embedding α β
    g : Finsupp β M
    hg : HasSubset.Subset (↑g.support) (Set.range ⇑f)
    ⊢ Eq (Finsupp.embDomain f (Finsupp.comapDomain (⇑f) g ⋯)) g
  -/
  ext b
  /-
    case h
    α : Type u_1
    β : Type u_2
    M : Type u_5
    inst✝ : Zero M
    f : Function.Embedding α β
    g : Finsupp β M
    hg : HasSubset.Subset (↑g.support) (Set.range ⇑f)
    b : β
    ⊢ Eq ((Finsupp.embDomain f (Finsupp.comapDomain (⇑f) g ⋯)) b) (g b)
  -/
  by_cases hb : b ∈ Set.range f
    /-
      case pos
      α : Type u_1
      β : Type u_2
      M : Type u_5
      inst✝ : Zero M
      f : Function.Embedding α β
      g : Finsupp β M
      hg : HasSubset.Subset (↑g.support) (Set.range ⇑f)
      b : β
      hb : Membership.mem (Set.range ⇑f) b
      ⊢ Eq ((Finsupp.embDomain f (Finsupp.comapDomain (⇑f) g ⋯)) b) (g b)
    -/
  · obtain ⟨a, rfl⟩ := hb
    /-
      case pos.intro
      α : Type u_1
      β : Type u_2
      M : Type u_5
      inst✝ : Zero M
      f : Function.Embedding α β
      g : Finsupp β M
      hg : HasSubset.Subset (↑g.support) (Set.range ⇑f)
      a : α
      ⊢ Eq ((Finsupp.embDomain f (Finsupp.comapDomain (⇑f) g ⋯)) (f a)) (g (f a))
    -/
    rw [embDomain_apply, comapDomain_apply]
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      β : Type u_2
      M : Type u_5
      inst✝ : Zero M
      f : Function.Embedding α β
      g : Finsupp β M
      hg : HasSubset.Subset (↑g.support) (Set.range ⇑f)
      b : β
      hb : Not (Membership.mem (Set.range ⇑f) b)
      ⊢ Eq ((Finsupp.embDomain f (Finsupp.comapDomain (⇑f) g ⋯)) b) (g b)
    -/
  · replace hg : g b = 0 := not_mem_support_iff.mp <| mt (hg ·) hb
    /-
      case neg
      α : Type u_1
      β : Type u_2
      M : Type u_5
      inst✝ : Zero M
      f : Function.Embedding α β
      g : Finsupp β M
      b : β
      hb : Not (Membership.mem (Set.range ⇑f) b)
      hg : Eq (g b) 0
      ⊢ Eq ((Finsupp.embDomain f (Finsupp.comapDomain (⇑f) g ⋯)) b) (g b)
    -/
    rw [embDomain_notin_range _ _ _ hb, hg]
    /-
      🎉 no goals
    -/


/-- Note the `hif` argument is needed for this to work in `rw`. -/
@[simp]
theorem comapDomain_zero (f : α → β)
    (hif : Set.InjOn f (f ⁻¹' ↑(0 : β →₀ M).support) := Finset.coe_empty ▸ (Set.injOn_empty f)) :
    comapDomain f (0 : β →₀ M) hif = (0 : α →₀ M) := by
  /-
    α : Type u_1
    β : Type u_2
    M : Type u_5
    inst✝ : Zero M
    f : α → β
    hif : optParam (Set.InjOn f (Set.preimage f ↑(Finsupp.support 0))) ⋯
    ⊢ Eq (Finsupp.comapDomain f 0 hif) 0
  -/
  ext
  /-
    case h
    α : Type u_1
    β : Type u_2
    M : Type u_5
    inst✝ : Zero M
    f : α → β
    hif : optParam (Set.InjOn f (Set.preimage f ↑(Finsupp.support 0))) ⋯
    a✝ : α
    ⊢ Eq ((Finsupp.comapDomain f 0 hif) a✝) (0 a✝)
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem comapDomain_single (f : α → β) (a : α) (m : M)
    (hif : Set.InjOn f (f ⁻¹' (single (f a) m).support)) :
    comapDomain f (Finsupp.single (f a) m) hif = Finsupp.single a m := by
  /-
    α : Type u_1
    β : Type u_2
    M : Type u_5
    inst✝ : Zero M
    f : α → β
    a : α
    m : M
    hif : Set.InjOn f (Set.preimage f ↑(Finsupp.single (f a) m).support)
    ⊢ Eq (Finsupp.comapDomain f (Finsupp.single (f a) m) hif) (Finsupp.single a m)
  -/
  rcases eq_or_ne m 0 with (rfl | hm)
    /-
      case inl
      α : Type u_1
      β : Type u_2
      M : Type u_5
      inst✝ : Zero M
      f : α → β
      a : α
      hif : Set.InjOn f (Set.preimage f ↑(Finsupp.single (f a) 0).support)
      ⊢ Eq (Finsupp.comapDomain f (Finsupp.single (f a) 0) hif) (Finsupp.single a 0)
    -/
  · simp only [single_zero, comapDomain_zero]
    /-
      🎉 no goals
    -/
  · rw [eq_single_iff, comapDomain_apply, comapDomain_support, ← Finset.coe_subset, coe_preimage,
      support_single_ne_zero _ hm, coe_singleton, coe_singleton, single_eq_same]
    /-
      case inr
      α : Type u_1
      β : Type u_2
      M : Type u_5
      inst✝ : Zero M
      f : α → β
      a : α
      m : M
      hif : Set.InjOn f (Set.preimage f ↑(Finsupp.single (f a) m).support)
      hm : Ne m 0
      ⊢ And (HasSubset.Subset (Set.preimage f (Singleton.singleton (f a))) (Singleto …
    -/
    rw [support_single_ne_zero _ hm, coe_singleton] at hif
    /-
      case inr
      α : Type u_1
      β : Type u_2
      M : Type u_5
      inst✝ : Zero M
      f : α → β
      a : α
      m : M
      hif : Set.InjOn f (Set.preimage f (Singleton.singleton (f a)))
      hm : Ne m 0
      ⊢ And (HasSubset.Subset (Set.preimage f (Singleton.singleton (f a))) (Singleto …
    -/
    exact ⟨fun x hx => hif hx rfl hx, rfl⟩
    /-
      🎉 no goals
    -/


theorem comapDomain_add (v₁ v₂ : β →₀ M) (hv₁ : Set.InjOn f (f ⁻¹' ↑v₁.support))
    (hv₂ : Set.InjOn f (f ⁻¹' ↑v₂.support)) (hv₁₂ : Set.InjOn f (f ⁻¹' ↑(v₁ + v₂).support)) :
    comapDomain f (v₁ + v₂) hv₁₂ = comapDomain f v₁ hv₁ + comapDomain f v₂ hv₂ := by
  /-
    α : Type u_1
    β : Type u_2
    M : Type u_5
    inst✝ : AddZeroClass M
    f : α → β
    v₁ v₂ : Finsupp β M
    hv₁ : Set.InjOn f (Set.preimage f ↑v₁.support)
    hv₂ : Set.InjOn f (Set.preimage f ↑v₂.support)
    hv₁₂ : Set.InjOn f (Set.preimage f ↑(HAdd.hAdd v₁ v₂).support)
    ⊢ Eq (Finsupp.comapDomain f (HAdd.hAdd v₁ v₂) hv₁₂) (HAdd.hAdd (Finsupp.comapD …
  -/
  ext
  /-
    case h
    α : Type u_1
    β : Type u_2
    M : Type u_5
    inst✝ : AddZeroClass M
    f : α → β
    v₁ v₂ : Finsupp β M
    hv₁ : Set.InjOn f (Set.preimage f ↑v₁.support)
    hv₂ : Set.InjOn f (Set.preimage f ↑v₂.support)
    hv₁₂ : Set.InjOn f (Set.preimage f ↑(HAdd.hAdd v₁ v₂).support)
    a✝ : α
    ⊢ Eq ((Finsupp.comapDomain f (HAdd.hAdd v₁ v₂) hv₁₂) a✝) ((HAdd.hAdd (Finsupp. …
  -/
  simp only [comapDomain_apply, coe_add, Pi.add_apply]
  /-
    🎉 no goals
  -/


/-- A version of `Finsupp.comapDomain_add` that's easier to use. -/
theorem comapDomain_add_of_injective (hf : Function.Injective f) (v₁ v₂ : β →₀ M) :
    comapDomain f (v₁ + v₂) hf.injOn =
      comapDomain f v₁ hf.injOn + comapDomain f v₂ hf.injOn :=
  comapDomain_add _ _ _ _ _


/-- `Finsupp.comapDomain` is an `AddMonoidHom`. -/
@[simps]
def comapDomain.addMonoidHom (hf : Function.Injective f) : (β →₀ M) →+ α →₀ M where
  toFun x := comapDomain f x hf.injOn
  map_zero' := comapDomain_zero f
  map_add' := comapDomain_add_of_injective hf


theorem mapDomain_comapDomain (hf : Function.Injective f) (l : β →₀ M)
    (hl : ↑l.support ⊆ Set.range f) :
    mapDomain f (comapDomain f l hf.injOn) = l := by
  /-
    α : Type u_1
    β : Type u_2
    M : Type u_5
    inst✝ : AddCommMonoid M
    f : α → β
    hf : Function.Injective f
    l : Finsupp β M
    hl : HasSubset.Subset (↑l.support) (Set.range f)
    ⊢ Eq (Finsupp.mapDomain f (Finsupp.comapDomain f l ⋯)) l
  -/
  conv_rhs => rw [← embDomain_comapDomain (f := ⟨f, hf⟩) hl (M := M), embDomain_eq_mapDomain]
  /-
    α : Type u_1
    β : Type u_2
    M : Type u_5
    inst✝ : AddCommMonoid M
    f : α → β
    hf : Function.Injective f
    l : Finsupp β M
    hl : HasSubset.Subset (↑l.support) (Set.range f)
    ⊢ Eq (Finsupp.mapDomain f (Finsupp.comapDomain f l ⋯)) (Finsupp.mapDomain (⇑{  …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- Restrict a finitely supported function on `Option α` to a finitely supported function on `α`. -/
def some [Zero M] (f : Option α →₀ M) : α →₀ M :=
                                        /-
                                          α : Type u_1
                                          β : Type u_2
                                          γ : Type u_3
                                          ι : Type u_4
                                          M : Type u_5
                                          M' : Type u_6
                                          N : Type u_7
                                          P : Type u_8
                                          G : Type u_9
                                          H : Type u_10
                                          R : Type u_11
                                          S : Type u_12
                                          inst✝ : Zero M
                                          f : Finsupp (Option α) M
                                          x✝ : α
                                          ⊢ Membership.mem (Set.preimage Option.some ↑f.support) x✝ → ∀ ⦃x₂ : α⦄, Member …
                                        -/
  f.comapDomain Option.some fun _ => by simp
                                        /-
                                          🎉 no goals
                                        -/


@[simp]
theorem some_apply [Zero M] (f : Option α →₀ M) (a : α) : f.some a = f (Option.some a) :=
  rfl


@[simp]
theorem some_zero [Zero M] : (0 : Option α →₀ M).some = 0 := by
  /-
    α : Type u_1
    M : Type u_5
    inst✝ : Zero M
    ⊢ Eq (Finsupp.some 0) 0
  -/
  ext
  /-
    case h
    α : Type u_1
    M : Type u_5
    inst✝ : Zero M
    a✝ : α
    ⊢ Eq ((Finsupp.some 0) a✝) (0 a✝)
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem some_add [AddCommMonoid M] (f g : Option α →₀ M) : (f + g).some = f.some + g.some := by
  /-
    α : Type u_1
    M : Type u_5
    inst✝ : AddCommMonoid M
    f g : Finsupp (Option α) M
    ⊢ Eq (HAdd.hAdd f g).some (HAdd.hAdd f.some g.some)
  -/
  ext
  /-
    case h
    α : Type u_1
    M : Type u_5
    inst✝ : AddCommMonoid M
    f g : Finsupp (Option α) M
    a✝ : α
    ⊢ Eq ((HAdd.hAdd f g).some a✝) ((HAdd.hAdd f.some g.some) a✝)
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem some_single_none [Zero M] (m : M) : (single none m : Option α →₀ M).some = 0 := by
  /-
    α : Type u_1
    M : Type u_5
    inst✝ : Zero M
    m : M
    ⊢ Eq (Finsupp.single Option.none m).some 0
  -/
  ext
  /-
    case h
    α : Type u_1
    M : Type u_5
    inst✝ : Zero M
    m : M
    a✝ : α
    ⊢ Eq ((Finsupp.single Option.none m).some a✝) (0 a✝)
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem some_single_some [Zero M] (a : α) (m : M) :
    (single (Option.some a) m : Option α →₀ M).some = single a m := by
  classical
    ext b
    simp [single_apply]


@[to_additive]
theorem prod_option_index [AddCommMonoid M] [CommMonoid N] (f : Option α →₀ M)
    (b : Option α → M → N) (h_zero : ∀ o, b o 0 = 1)
    (h_add : ∀ o m₁ m₂, b o (m₁ + m₂) = b o m₁ * b o m₂) :
    f.prod b = b none (f none) * f.some.prod fun a => b (Option.some a) := by
  classical
    apply induction_linear f
    · simp [some_zero, h_zero]
    · intro f₁ f₂ h₁ h₂
      rw [Finsupp.prod_add_index, h₁, h₂, some_add, Finsupp.prod_add_index]
      · simp only [h_add, Pi.add_apply, Finsupp.coe_add]
        rw [mul_mul_mul_comm]
      all_goals simp [h_zero, h_add]
    · rintro (_ | a) m <;> simp [h_zero, h_add]


theorem sum_option_index_smul [Semiring R] [AddCommMonoid M] [Module R M] (f : Option α →₀ R)
    (b : Option α → M) :
    (f.sum fun o r => r • b o) = f none • b none + f.some.sum fun a r => r • b (Option.some a) :=
  f.sum_option_index _ (fun _ => zero_smul _ _) fun _ _ _ => add_smul _ _ _


/--
`Finsupp.filter p f` is the finitely supported function that is `f a` if `p a` is true and `0`
otherwise. -/
def filter (p : α → Prop) [DecidablePred p] (f : α →₀ M) : α →₀ M where
  toFun a := if p a then f a else 0
  support := f.support.filter p
  mem_support_toFun a := by
    /-
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      ι : Type u_4
      M : Type u_5
      M' : Type u_6
      N : Type u_7
      P : Type u_8
      G : Type u_9
      H : Type u_10
      R : Type u_11
      S : Type u_12
      inst✝² : Zero M
      p✝ : α → Prop
      inst✝¹ : DecidablePred p✝
      f✝ : Finsupp α M
      p : α → Prop
      inst✝ : DecidablePred p
      f : Finsupp α M
      a : α
      ⊢ Iff (Membership.mem (Finset.filter p f.support) a) (Ne ((fun a => ite (p a)  …
    -/
    beta_reduce -- Porting note (https://github.com/leanprover-community/mathlib4/issues/12129): additional beta reduction needed to activate `split_ifs`
    /-
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      ι : Type u_4
      M : Type u_5
      M' : Type u_6
      N : Type u_7
      P : Type u_8
      G : Type u_9
      H : Type u_10
      R : Type u_11
      S : Type u_12
      inst✝² : Zero M
      p✝ : α → Prop
      inst✝¹ : DecidablePred p✝
      f✝ : Finsupp α M
      p : α → Prop
      inst✝ : DecidablePred p
      f : Finsupp α M
      a : α
      ⊢ Iff (Membership.mem (Finset.filter p f.support) a) (Ne (ite (p a) (f a) 0) 0)
    -/
    split_ifs with h <;>
        /-
          case pos
          α : Type u_1
          β : Type u_2
          γ : Type u_3
          ι : Type u_4
          M : Type u_5
          M' : Type u_6
          N : Type u_7
          P : Type u_8
          G : Type u_9
          H : Type u_10
          R : Type u_11
          S : Type u_12
          inst✝² : Zero M
          p✝ : α → Prop
          inst✝¹ : DecidablePred p✝
          f✝ : Finsupp α M
          p : α → Prop
          inst✝ : DecidablePred p
          f : Finsupp α M
          a : α
          h : p a
          ⊢ Iff (Membership.mem (Finset.filter p f.support) a) (Ne (f a) 0)
        -/
        /-
          case pos
          α : Type u_1
          β : Type u_2
          γ : Type u_3
          ι : Type u_4
          M : Type u_5
          M' : Type u_6
          N : Type u_7
          P : Type u_8
          G : Type u_9
          H : Type u_10
          R : Type u_11
          S : Type u_12
          inst✝² : Zero M
          p✝ : α → Prop
          inst✝¹ : DecidablePred p✝
          f✝ : Finsupp α M
          p : α → Prop
          inst✝ : DecidablePred p
          f : Finsupp α M
          a : α
          h : p a
          ⊢ Iff (And (Ne (f a) 0) True) (Ne (f a) 0)
        -/
        /-
          🎉 no goals
        -/
        /-
          case neg
          α : Type u_1
          β : Type u_2
          γ : Type u_3
          ι : Type u_4
          M : Type u_5
          M' : Type u_6
          N : Type u_7
          P : Type u_8
          G : Type u_9
          H : Type u_10
          R : Type u_11
          S : Type u_12
          inst✝² : Zero M
          p✝ : α → Prop
          inst✝¹ : DecidablePred p✝
          f✝ : Finsupp α M
          p : α → Prop
          inst✝ : DecidablePred p
          f : Finsupp α M
          a : α
          h : Not (p a)
          ⊢ Iff (And (Ne (f a) 0) False) (Ne 0 0)
        -/
        tauto
        /-
          🎉 no goals
        -/


theorem filter_apply (a : α) : f.filter p a = if p a then f a else 0 := rfl


theorem filter_eq_indicator : ⇑(f.filter p) = Set.indicator { x | p x } f := by
  /-
    α : Type u_1
    M : Type u_5
    inst✝¹ : Zero M
    p : α → Prop
    inst✝ : DecidablePred p
    f : Finsupp α M
    ⊢ Eq (⇑(Finsupp.filter p f)) ((setOf fun x => p x).indicator ⇑f)
  -/
  ext
  /-
    case h
    α : Type u_1
    M : Type u_5
    inst✝¹ : Zero M
    p : α → Prop
    inst✝ : DecidablePred p
    f : Finsupp α M
    x✝ : α
    ⊢ Eq ((Finsupp.filter p f) x✝) ((setOf fun x => p x).indicator (⇑f) x✝)
  -/
  simp [filter_apply, Set.indicator_apply]
  /-
    🎉 no goals
  -/


theorem filter_eq_zero_iff : f.filter p = 0 ↔ ∀ x, p x → f x = 0 := by
  simp only [DFunLike.ext_iff, filter_eq_indicator, zero_apply, Set.indicator_apply_eq_zero,
    Set.mem_setOf_eq]


theorem filter_eq_self_iff : f.filter p = f ↔ ∀ x, f x ≠ 0 → p x := by
  simp only [DFunLike.ext_iff, filter_eq_indicator, Set.indicator_apply_eq_self, Set.mem_setOf_eq,
    not_imp_comm]


@[simp]
theorem filter_apply_pos {a : α} (h : p a) : f.filter p a = f a := if_pos h


@[simp]
theorem filter_apply_neg {a : α} (h : ¬p a) : f.filter p a = 0 := if_neg h


@[simp]
theorem support_filter : (f.filter p).support = {x ∈ f.support | p x} := rfl


theorem filter_zero : (0 : α →₀ M).filter p = 0 := by
  /-
    α : Type u_1
    M : Type u_5
    inst✝¹ : Zero M
    p : α → Prop
    inst✝ : DecidablePred p
    ⊢ Eq (Finsupp.filter p 0) 0
  -/
  classical rw [← support_eq_empty, support_filter, support_zero, Finset.filter_empty]
  /-
    🎉 no goals
  -/


@[simp]
theorem filter_single_of_pos {a : α} {b : M} (h : p a) : (single a b).filter p = single a b :=
  (filter_eq_self_iff _ _).2 fun _ hx => (single_apply_ne_zero.1 hx).1.symm ▸ h


@[simp]
theorem filter_single_of_neg {a : α} {b : M} (h : ¬p a) : (single a b).filter p = 0 :=
  (filter_eq_zero_iff _ _).2 fun _ hpx =>
    single_apply_eq_zero.2 fun hxa => absurd hpx (hxa.symm ▸ h)


@[to_additive]
theorem prod_filter_index [CommMonoid N] (g : α → M → N) :
    (f.filter p).prod g = ∏ x ∈ (f.filter p).support, g x (f x) := by
  classical
    refine Finset.prod_congr rfl fun x hx => ?_
    rw [support_filter, Finset.mem_filter] at hx
    rw [filter_apply_pos _ _ hx.2]


@[to_additive (attr := simp)]
theorem prod_filter_mul_prod_filter_not [CommMonoid N] (g : α → M → N) :
    (f.filter p).prod g * (f.filter fun a => ¬p a).prod g = f.prod g := by
  classical simp_rw [prod_filter_index, support_filter, Finset.prod_filter_mul_prod_filter_not,
    Finsupp.prod]


@[to_additive (attr := simp)]
theorem prod_div_prod_filter [CommGroup G] (g : α → M → G) :
    f.prod g / (f.filter p).prod g = (f.filter fun a => ¬p a).prod g :=
  div_eq_of_eq_mul' (prod_filter_mul_prod_filter_not _ _ _).symm


theorem filter_pos_add_filter_neg [AddZeroClass M] (f : α →₀ M) (p : α → Prop) [DecidablePred p] :
    (f.filter p + f.filter fun a => ¬p a) = f :=
  DFunLike.coe_injective <| by
    /-
      α : Type u_1
      M : Type u_5
      inst✝¹ : AddZeroClass M
      f : Finsupp α M
      p : α → Prop
      inst✝ : DecidablePred p
      ⊢ Eq ((fun f => ⇑f) (HAdd.hAdd (Finsupp.filter p f) (Finsupp.filter (fun a =>  …
    -/
    simp only [coe_add, filter_eq_indicator]
    /-
      α : Type u_1
      M : Type u_5
      inst✝¹ : AddZeroClass M
      f : Finsupp α M
      p : α → Prop
      inst✝ : DecidablePred p
      ⊢ Eq (HAdd.hAdd ((setOf fun x => p x).indicator ⇑f) ((setOf fun a => Not (p a) …
    -/
    exact Set.indicator_self_add_compl { x | p x } f
    /-
      🎉 no goals
    -/


/-- `frange f` is the image of `f` on the support of `f`. -/
def frange (f : α →₀ M) : Finset M :=
  haveI := Classical.decEq M
  Finset.image f f.support


theorem mem_frange {f : α →₀ M} {y : M} : y ∈ f.frange ↔ y ≠ 0 ∧ ∃ x, f x = y := by
  /-
    α : Type u_1
    M : Type u_5
    inst✝ : Zero M
    f : Finsupp α M
    y : M
    ⊢ Iff (Membership.mem f.frange y) (And (Ne y 0) (Exists fun x => Eq (f x) y))
  -/
  rw [frange, @Finset.mem_image _ _ (Classical.decEq _) _ f.support]
  exact ⟨fun ⟨x, hx1, hx2⟩ => ⟨hx2 ▸ mem_support_iff.1 hx1, x, hx2⟩, fun ⟨hy, x, hx⟩ =>
    ⟨x, mem_support_iff.2 (hx.symm ▸ hy), hx⟩⟩
  -- Porting note: maybe there is a better way to fix this, but (1) it wasn't seeing past `frange`
  -- the definition, and (2) it needed the `Classical.decEq` instance again.


theorem zero_not_mem_frange {f : α →₀ M} : (0 : M) ∉ f.frange := fun H => (mem_frange.1 H).1 rfl


theorem frange_single {x : α} {y : M} : frange (single x y) ⊆ {y} := fun r hr =>
  let ⟨t, ht1, ht2⟩ := mem_frange.1 hr
  ht2 ▸ by
    classical
      rw [single_apply] at ht2 ⊢
      split_ifs at ht2 ⊢
      · exact Finset.mem_singleton_self _
      · exact (t ht2.symm).elim


/--
`subtypeDomain p f` is the restriction of the finitely supported function `f` to subtype `p`. -/
def subtypeDomain (p : α → Prop) (f : α →₀ M) : Subtype p →₀ M where
  support :=
    haveI := Classical.decPred p
    f.support.subtype p
  toFun := f ∘ Subtype.val
                            /-
                              α : Type u_1
                              β : Type u_2
                              γ : Type u_3
                              ι : Type u_4
                              M : Type u_5
                              M' : Type u_6
                              N : Type u_7
                              P : Type u_8
                              G : Type u_9
                              H : Type u_10
                              R : Type u_11
                              S : Type u_12
                              inst✝ : Zero M
                              p✝ p : α → Prop
                              f : Finsupp α M
                              a : Subtype p
                              ⊢ Iff (Membership.mem (Finset.subtype p f.support) a) (Ne (Function.comp (⇑f)  …
                            -/
  mem_support_toFun a := by simp only [@mem_subtype _ _ (Classical.decPred p), mem_support_iff]; rfl
                                                                                                 /-
                                                                                                   🎉 no goals
                                                                                                 -/


@[simp]
theorem support_subtypeDomain [D : DecidablePred p] {f : α →₀ M} :
                                                            /-
                                                              α : Type u_1
                                                              M : Type u_5
                                                              inst✝ : Zero M
                                                              p : α → Prop
                                                              D : DecidablePred p
                                                              f : Finsupp α M
                                                              ⊢ Eq (Finsupp.subtypeDomain p f).support (Finset.subtype p f.support)
                                                            -/
    (subtypeDomain p f).support = f.support.subtype p := by rw [Subsingleton.elim D] <;> rfl
                                                                                         /-
                                                                                           🎉 no goals
                                                                                         -/


@[simp]
theorem subtypeDomain_apply {a : Subtype p} {v : α →₀ M} : (subtypeDomain p v) a = v a.val :=
  rfl


@[simp]
theorem subtypeDomain_zero : subtypeDomain p (0 : α →₀ M) = 0 :=
  rfl


theorem subtypeDomain_eq_zero_iff' {f : α →₀ M} : f.subtypeDomain p = 0 ↔ ∀ x, p x → f x = 0 := by
  classical simp_rw [← support_eq_empty, support_subtypeDomain, subtype_eq_empty,
      not_mem_support_iff]


theorem subtypeDomain_eq_zero_iff {f : α →₀ M} (hf : ∀ x ∈ f.support, p x) :
    f.subtypeDomain p = 0 ↔ f = 0 :=
  subtypeDomain_eq_zero_iff'.trans
    ⟨fun H =>
      ext fun x => by
        /-
          α : Type u_1
          M : Type u_5
          inst✝ : Zero M
          p : α → Prop
          f : Finsupp α M
          hf : ∀ (x : α), Membership.mem f.support x → p x
          H : ∀ (x : α), p x → Eq (f x) 0
          x : α
          ⊢ Eq (f x) (0 x)
        -/
        classical exact if hx : p x then H x hx else not_mem_support_iff.1 <| mt (hf x) hx,
        /-
          🎉 no goals
        -/
                      /-
                        α : Type u_1
                        M : Type u_5
                        inst✝ : Zero M
                        p : α → Prop
                        f : Finsupp α M
                        hf : ∀ (x : α), Membership.mem f.support x → p x
                        H : Eq f 0
                        x : α
                        x✝ : p x
                        ⊢ Eq (f x) 0
                      -/
      fun H x _ => by simp [H]⟩
                      /-
                        🎉 no goals
                      -/


@[to_additive]
theorem prod_subtypeDomain_index [CommMonoid N] {v : α →₀ M} {h : α → M → N}
    (hp : ∀ x ∈ v.support, p x) : (v.subtypeDomain p).prod (fun a b ↦ h a b) = v.prod h := by
  /-
    α : Type u_1
    M : Type u_5
    N : Type u_7
    inst✝¹ : Zero M
    p : α → Prop
    inst✝ : CommMonoid N
    v : Finsupp α M
    h : α → M → N
    hp : ∀ (x : α), Membership.mem v.support x → p x
    ⊢ Eq ((Finsupp.subtypeDomain p v).prod fun a b => h (↑a) b) (v.prod h)
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
  refine Finset.prod_bij (fun p _ ↦ p) ?_ ?_ ?_ ?_ <;> aesop
                                                       /-
                                                         🎉 no goals
                                                       -/


@[simp]
theorem subtypeDomain_add {v v' : α →₀ M} :
    (v + v').subtypeDomain p = v.subtypeDomain p + v'.subtypeDomain p :=
  ext fun _ => rfl


/-- `subtypeDomain` but as an `AddMonoidHom`. -/
def subtypeDomainAddMonoidHom : (α →₀ M) →+ Subtype p →₀ M where
  toFun := subtypeDomain p
  map_zero' := subtypeDomain_zero
  map_add' _ _ := subtypeDomain_add


/-- `Finsupp.filter` as an `AddMonoidHom`. -/
def filterAddHom (p : α → Prop) [DecidablePred p] : (α →₀ M) →+ α →₀ M where
  toFun := filter p
  map_zero' := filter_zero p
  map_add' f g := DFunLike.coe_injective <| by
    /-
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      ι : Type u_4
      M : Type u_5
      M' : Type u_6
      N : Type u_7
      P : Type u_8
      G : Type u_9
      H : Type u_10
      R : Type u_11
      S : Type u_12
      inst✝¹ : AddZeroClass M
      p✝ : α → Prop
      v v' : Finsupp α M
      p : α → Prop
      inst✝ : DecidablePred p
      f g : Finsupp α M
      ⊢ Eq ((fun f => ⇑f) ({ toFun := Finsupp.filter p, map_zero' := ⋯ }.toFun (HAdd …
    -/
    simp only [filter_eq_indicator, coe_add]
    /-
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      ι : Type u_4
      M : Type u_5
      M' : Type u_6
      N : Type u_7
      P : Type u_8
      G : Type u_9
      H : Type u_10
      R : Type u_11
      S : Type u_12
      inst✝¹ : AddZeroClass M
      p✝ : α → Prop
      v v' : Finsupp α M
      p : α → Prop
      inst✝ : DecidablePred p
      f g : Finsupp α M
      ⊢ Eq ((setOf fun x => p x).indicator (HAdd.hAdd ⇑f ⇑g)) (HAdd.hAdd ((setOf fun …
    -/
    exact Set.indicator_add { x | p x } f g
    /-
      🎉 no goals
    -/


@[simp]
theorem filter_add [DecidablePred p] {v v' : α →₀ M} :
    (v + v').filter p = v.filter p + v'.filter p :=
  (filterAddHom p).map_add v v'


theorem subtypeDomain_sum {s : Finset ι} {h : ι → α →₀ M} :
    (∑ c ∈ s, h c).subtypeDomain p = ∑ c ∈ s, (h c).subtypeDomain p :=
  map_sum subtypeDomainAddMonoidHom _ s


theorem subtypeDomain_finsupp_sum [Zero N] {s : β →₀ N} {h : β → N → α →₀ M} :
    (s.sum h).subtypeDomain p = s.sum fun c d => (h c d).subtypeDomain p :=
  subtypeDomain_sum


theorem filter_sum [DecidablePred p] (s : Finset ι) (f : ι → α →₀ M) :
    (∑ a ∈ s, f a).filter p = ∑ a ∈ s, filter p (f a) :=
  map_sum (filterAddHom p) f s


theorem filter_eq_sum (p : α → Prop) [DecidablePred p] (f : α →₀ M) :
    f.filter p = ∑ i ∈ f.support.filter p, single i (f i) :=
  (f.filter p).sum_single.symm.trans <|
    Finset.sum_congr rfl fun x hx => by
      /-
        α : Type u_1
        M : Type u_5
        inst✝¹ : AddCommMonoid M
        p : α → Prop
        inst✝ : DecidablePred p
        f : Finsupp α M
        x : α
        hx : Membership.mem (Finset.filter p f.support) x
        ⊢ Eq (Finsupp.single x ((Finsupp.filter p f) x)) (Finsupp.single x (f x))
      -/
      rw [filter_apply_pos _ _ (mem_filter.1 hx).2]
      /-
        🎉 no goals
      -/


@[simp]
theorem subtypeDomain_neg : (-v).subtypeDomain p = -v.subtypeDomain p :=
  ext fun _ => rfl


@[simp]
theorem subtypeDomain_sub : (v - v').subtypeDomain p = v.subtypeDomain p - v'.subtypeDomain p :=
  ext fun _ => rfl


@[simp]
theorem single_neg (a : α) (b : G) : single a (-b) = -single a b :=
  (singleAddHom a : G →+ _).map_neg b


@[simp]
theorem single_sub (a : α) (b₁ b₂ : G) : single a (b₁ - b₂) = single a b₁ - single a b₂ :=
  (singleAddHom a : G →+ _).map_sub b₁ b₂


@[simp]
theorem erase_neg (a : α) (f : α →₀ G) : erase a (-f) = -erase a f :=
  (eraseAddHom a : (_ →₀ G) →+ _).map_neg f


@[simp]
theorem erase_sub (a : α) (f₁ f₂ : α →₀ G) : erase a (f₁ - f₂) = erase a f₁ - erase a f₂ :=
  (eraseAddHom a : (_ →₀ G) →+ _).map_sub f₁ f₂


@[simp]
theorem filter_neg (p : α → Prop) [DecidablePred p] (f : α →₀ G) : filter p (-f) = -filter p f :=
  (filterAddHom p : (_ →₀ G) →+ _).map_neg f


@[simp]
theorem filter_sub (p : α → Prop) [DecidablePred p] (f₁ f₂ : α →₀ G) :
    filter p (f₁ - f₂) = filter p f₁ - filter p f₂ :=
  (filterAddHom p : (_ →₀ G) →+ _).map_sub f₁ f₂


theorem mem_support_multiset_sum [AddCommMonoid M] {s : Multiset (α →₀ M)} (a : α) :
    a ∈ s.sum.support → ∃ f ∈ s, a ∈ (f : α →₀ M).support :=
                                                   /-
                                                     α : Type u_1
                                                     M : Type u_5
                                                     inst✝ : AddCommMonoid M
                                                     s : Multiset (Finsupp α M)
                                                     a : α
                                                     h : Membership.mem (Multiset.sum 0).support a
                                                     ⊢ False
                                                   -/
  Multiset.induction_on s (fun h => False.elim (by simp at h))
                                                   /-
                                                     🎉 no goals
                                                   -/
    (by
      /-
        α : Type u_1
        M : Type u_5
        inst✝ : AddCommMonoid M
        s : Multiset (Finsupp α M)
        a : α
        ⊢ ∀ (a_1 : Finsupp α M) (s : Multiset (Finsupp α M)), (Membership.mem s.sum.su …
      -/
      intro f s ih ha
      /-
        α : Type u_1
        M : Type u_5
        inst✝ : AddCommMonoid M
        s✝ : Multiset (Finsupp α M)
        a : α
        f : Finsupp α M
        s : Multiset (Finsupp α M)
        ih : Membership.mem s.sum.support a → Exists fun f => And (Membership.mem s f) …
        ha : Membership.mem (Multiset.cons f s).sum.support a
        ⊢ Exists fun f_1 => And (Membership.mem (Multiset.cons f s) f_1) (Membership.m …
      -/
      by_cases h : a ∈ f.support
        /-
          case pos
          α : Type u_1
          M : Type u_5
          inst✝ : AddCommMonoid M
          s✝ : Multiset (Finsupp α M)
          a : α
          f : Finsupp α M
          s : Multiset (Finsupp α M)
          ih : Membership.mem s.sum.support a → Exists fun f => And (Membership.mem s f) …
          ha : Membership.mem (Multiset.cons f s).sum.support a
          h : Membership.mem f.support a
          ⊢ Exists fun f_1 => And (Membership.mem (Multiset.cons f s) f_1) (Membership.m …
        -/
      · exact ⟨f, Multiset.mem_cons_self _ _, h⟩
        /-
          🎉 no goals
        -/
      · simp only [Multiset.sum_cons, mem_support_iff, add_apply, not_mem_support_iff.1 h,
          zero_add] at ha
        /-
          case neg
          α : Type u_1
          M : Type u_5
          inst✝ : AddCommMonoid M
          s✝ : Multiset (Finsupp α M)
          a : α
          f : Finsupp α M
          s : Multiset (Finsupp α M)
          ih : Membership.mem s.sum.support a → Exists fun f => And (Membership.mem s f) …
          h : Not (Membership.mem f.support a)
          ha : Ne (s.sum a) 0
          ⊢ Exists fun f_1 => And (Membership.mem (Multiset.cons f s) f_1) (Membership.m …
        -/
        rcases ih (mem_support_iff.2 ha) with ⟨f', h₀, h₁⟩
        /-
          case neg.intro.intro
          α : Type u_1
          M : Type u_5
          inst✝ : AddCommMonoid M
          s✝ : Multiset (Finsupp α M)
          a : α
          f : Finsupp α M
          s : Multiset (Finsupp α M)
          ih : Membership.mem s.sum.support a → Exists fun f => And (Membership.mem s f) …
          h : Not (Membership.mem f.support a)
          ha : Ne (s.sum a) 0
          f' : Finsupp α M
          h₀ : Membership.mem s f'
          h₁ : Membership.mem f'.support a
          ⊢ Exists fun f_1 => And (Membership.mem (Multiset.cons f s) f_1) (Membership.m …
        -/
        exact ⟨f', Multiset.mem_cons_of_mem h₀, h₁⟩)
        /-
          🎉 no goals
        -/


theorem mem_support_finset_sum [AddCommMonoid M] {s : Finset ι} {h : ι → α →₀ M} (a : α)
    (ha : a ∈ (∑ c ∈ s, h c).support) : ∃ c ∈ s, a ∈ (h c).support :=
  let ⟨_, hf, hfa⟩ := mem_support_multiset_sum a ha
  let ⟨c, hc, Eq⟩ := Multiset.mem_map.1 hf
  ⟨c, hc, Eq.symm ▸ hfa⟩


/-- Given a finitely supported function `f` from a product type `α × β` to `γ`,
`curry f` is the "curried" finitely supported function from `α` to the type of
finitely supported functions from `β` to `γ`. -/
protected def curry (f : α × β →₀ M) : α →₀ β →₀ M :=
  f.sum fun p c => single p.1 (single p.2 c)


@[simp]
theorem curry_apply (f : α × β →₀ M) (x : α) (y : β) : f.curry x y = f (x, y) := by
  classical
    have : ∀ b : α × β, single b.fst (single b.snd (f b)) x y = if b = (x, y) then f b else 0 := by
      rintro ⟨b₁, b₂⟩
      simp only [ne_eq, single_apply, Prod.ext_iff, ite_and]
      split_ifs <;> simp [single_apply, *]
    rw [Finsupp.curry, sum_apply, sum_apply, sum_eq_single, this, if_pos rfl]
    · intro b _ b_ne
      rw [this b, if_neg b_ne]
    · intro _
      rw [single_zero, single_zero, coe_zero, Pi.zero_apply, coe_zero, Pi.zero_apply]


theorem sum_curry_index (f : α × β →₀ M) (g : α → β → M → N) (hg₀ : ∀ a b, g a b 0 = 0)
    (hg₁ : ∀ a b c₀ c₁, g a b (c₀ + c₁) = g a b c₀ + g a b c₁) :
    (f.curry.sum fun a f => f.sum (g a)) = f.sum fun p c => g p.1 p.2 c := by
  /-
    α : Type u_1
    β : Type u_2
    M : Type u_5
    N : Type u_7
    inst✝¹ : AddCommMonoid M
    inst✝ : AddCommMonoid N
    f : Finsupp (Prod α β) M
    g : α → β → M → N
    hg₀ : ∀ (a : α) (b : β), Eq (g a b 0) 0
    hg₁ : ∀ (a : α) (b : β) (c₀ c₁ : M), Eq (g a b (HAdd.hAdd c₀ c₁)) (HAdd.hAdd ( …
    ⊢ Eq (f.curry.sum fun a f => f.sum (g a)) (f.sum fun p c => g p.1 p.2 c)
  -/
  rw [Finsupp.curry]
  /-
    α : Type u_1
    β : Type u_2
    M : Type u_5
    N : Type u_7
    inst✝¹ : AddCommMonoid M
    inst✝ : AddCommMonoid N
    f : Finsupp (Prod α β) M
    g : α → β → M → N
    hg₀ : ∀ (a : α) (b : β), Eq (g a b 0) 0
    hg₁ : ∀ (a : α) (b : β) (c₀ c₁ : M), Eq (g a b (HAdd.hAdd c₀ c₁)) (HAdd.hAdd ( …
    ⊢ Eq ((f.sum fun p c => Finsupp.single p.1 (Finsupp.single p.2 c)).sum fun a f …
  -/
  trans
  · exact
      sum_sum_index (fun a => sum_zero_index) fun a b₀ b₁ =>
        sum_add_index' (fun a => hg₀ _ _) fun c d₀ d₁ => hg₁ _ _ _ _
  /-
    α : Type u_1
    β : Type u_2
    M : Type u_5
    N : Type u_7
    inst✝¹ : AddCommMonoid M
    inst✝ : AddCommMonoid N
    f : Finsupp (Prod α β) M
    g : α → β → M → N
    hg₀ : ∀ (a : α) (b : β), Eq (g a b 0) 0
    hg₁ : ∀ (a : α) (b : β) (c₀ c₁ : M), Eq (g a b (HAdd.hAdd c₀ c₁)) (HAdd.hAdd ( …
    ⊢ Eq (f.sum fun a b => (Finsupp.single a.1 (Finsupp.single a.2 b)).sum fun a f …
  -/
  congr; funext p c
  /-
    case e_g.h.h
    α : Type u_1
    β : Type u_2
    M : Type u_5
    N : Type u_7
    inst✝¹ : AddCommMonoid M
    inst✝ : AddCommMonoid N
    f : Finsupp (Prod α β) M
    g : α → β → M → N
    hg₀ : ∀ (a : α) (b : β), Eq (g a b 0) 0
    hg₁ : ∀ (a : α) (b : β) (c₀ c₁ : M), Eq (g a b (HAdd.hAdd c₀ c₁)) (HAdd.hAdd ( …
    p : Prod α β
    c : M
    ⊢ Eq ((Finsupp.single p.1 (Finsupp.single p.2 c)).sum fun a f => f.sum (g a))  …
  -/
  trans
    /-
      α : Type u_1
      β : Type u_2
      M : Type u_5
      N : Type u_7
      inst✝¹ : AddCommMonoid M
      inst✝ : AddCommMonoid N
      f : Finsupp (Prod α β) M
      g : α → β → M → N
      hg₀ : ∀ (a : α) (b : β), Eq (g a b 0) 0
      hg₁ : ∀ (a : α) (b : β) (c₀ c₁ : M), Eq (g a b (HAdd.hAdd c₀ c₁)) (HAdd.hAdd ( …
      p : Prod α β
      c : M
      ⊢ Eq ((Finsupp.single p.1 (Finsupp.single p.2 c)).sum fun a f => f.sum (g a))  …
    -/
  · exact sum_single_index sum_zero_index
    /-
      🎉 no goals
    -/
  /-
    α : Type u_1
    β : Type u_2
    M : Type u_5
    N : Type u_7
    inst✝¹ : AddCommMonoid M
    inst✝ : AddCommMonoid N
    f : Finsupp (Prod α β) M
    g : α → β → M → N
    hg₀ : ∀ (a : α) (b : β), Eq (g a b 0) 0
    hg₁ : ∀ (a : α) (b : β) (c₀ c₁ : M), Eq (g a b (HAdd.hAdd c₀ c₁)) (HAdd.hAdd ( …
    p : Prod α β
    c : M
    ⊢ Eq ((Finsupp.single p.2 c).sum (g p.1)) (g p.1 p.2 c)
  -/
  exact sum_single_index (hg₀ _ _)
  /-
    🎉 no goals
  -/


/-- Given a finitely supported function `f` from `α` to the type of
finitely supported functions from `β` to `M`,
`uncurry f` is the "uncurried" finitely supported function from `α × β` to `M`. -/
protected def uncurry (f : α →₀ β →₀ M) : α × β →₀ M :=
  f.sum fun a g => g.sum fun b c => single (a, b) c


/-- `finsuppProdEquiv` defines the `Equiv` between `((α × β) →₀ M)` and `(α →₀ (β →₀ M))` given by
currying and uncurrying. -/
def finsuppProdEquiv : (α × β →₀ M) ≃ (α →₀ β →₀ M) where
  toFun := Finsupp.curry
  invFun := Finsupp.uncurry
  left_inv f := by
    /-
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      ι : Type u_4
      M : Type u_5
      M' : Type u_6
      N : Type u_7
      P : Type u_8
      G : Type u_9
      H : Type u_10
      R : Type u_11
      S : Type u_12
      inst✝¹ : AddCommMonoid M
      inst✝ : AddCommMonoid N
      f : Finsupp (Prod α β) M
      ⊢ Eq f.curry.uncurry f
    -/
    rw [Finsupp.uncurry, sum_curry_index]
      /-
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        ι : Type u_4
        M : Type u_5
        M' : Type u_6
        N : Type u_7
        P : Type u_8
        G : Type u_9
        H : Type u_10
        R : Type u_11
        S : Type u_12
        inst✝¹ : AddCommMonoid M
        inst✝ : AddCommMonoid N
        f : Finsupp (Prod α β) M
        ⊢ Eq (f.sum fun p c => Finsupp.single { fst := p.1, snd := p.2 } c) f
      -/
    · simp_rw [Prod.mk.eta, sum_single]
      /-
        🎉 no goals
      -/
      /-
        case hg₀
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        ι : Type u_4
        M : Type u_5
        M' : Type u_6
        N : Type u_7
        P : Type u_8
        G : Type u_9
        H : Type u_10
        R : Type u_11
        S : Type u_12
        inst✝¹ : AddCommMonoid M
        inst✝ : AddCommMonoid N
        f : Finsupp (Prod α β) M
        ⊢ ∀ (a : α) (b : β), Eq (Finsupp.single { fst := a, snd := b } 0) 0
      -/
    · intros
      /-
        case hg₀
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        ι : Type u_4
        M : Type u_5
        M' : Type u_6
        N : Type u_7
        P : Type u_8
        G : Type u_9
        H : Type u_10
        R : Type u_11
        S : Type u_12
        inst✝¹ : AddCommMonoid M
        inst✝ : AddCommMonoid N
        f : Finsupp (Prod α β) M
        a✝ : α
        b✝ : β
        ⊢ Eq (Finsupp.single { fst := a✝, snd := b✝ } 0) 0
      -/
      apply single_zero
      /-
        🎉 no goals
      -/
      /-
        case hg₁
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        ι : Type u_4
        M : Type u_5
        M' : Type u_6
        N : Type u_7
        P : Type u_8
        G : Type u_9
        H : Type u_10
        R : Type u_11
        S : Type u_12
        inst✝¹ : AddCommMonoid M
        inst✝ : AddCommMonoid N
        f : Finsupp (Prod α β) M
        ⊢ ∀ (a : α) (b : β) (c₀ c₁ : M), Eq (Finsupp.single { fst := a, snd := b } (HA …
      -/
    · intros
      /-
        case hg₁
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        ι : Type u_4
        M : Type u_5
        M' : Type u_6
        N : Type u_7
        P : Type u_8
        G : Type u_9
        H : Type u_10
        R : Type u_11
        S : Type u_12
        inst✝¹ : AddCommMonoid M
        inst✝ : AddCommMonoid N
        f : Finsupp (Prod α β) M
        a✝ : α
        b✝ : β
        c₀✝ c₁✝ : M
        ⊢ Eq (Finsupp.single { fst := a✝, snd := b✝ } (HAdd.hAdd c₀✝ c₁✝)) (HAdd.hAdd  …
      -/
      apply single_add
      /-
        🎉 no goals
      -/
  right_inv f := by
    simp only [Finsupp.curry, Finsupp.uncurry, sum_sum_index, sum_zero_index, sum_add_index,
      sum_single_index, single_zero, single_add, eq_self_iff_true, forall_true_iff,
      forall₃_true_iff, (single_sum _ _ _).symm, sum_single]


theorem filter_curry (f : α × β →₀ M) (p : α → Prop) [DecidablePred p] :
    (f.filter fun a : α × β => p a.1).curry = f.curry.filter p := by
  classical
    rw [Finsupp.curry, Finsupp.curry, Finsupp.sum, Finsupp.sum, filter_sum, support_filter,
      sum_filter]
    refine Finset.sum_congr rfl ?_
    rintro ⟨a₁, a₂⟩ _
    split_ifs with h
    · rw [filter_apply_pos, filter_single_of_pos] <;> exact h
    · rwa [filter_single_of_neg]


theorem support_curry [DecidableEq α] (f : α × β →₀ M) :
    f.curry.support ⊆ f.support.image Prod.fst := by
  /-
    α : Type u_1
    β : Type u_2
    M : Type u_5
    inst✝¹ : AddCommMonoid M
    inst✝ : DecidableEq α
    f : Finsupp (Prod α β) M
    ⊢ HasSubset.Subset f.curry.support (Finset.image Prod.fst f.support)
  -/
  rw [← Finset.biUnion_singleton]
  /-
    α : Type u_1
    β : Type u_2
    M : Type u_5
    inst✝¹ : AddCommMonoid M
    inst✝ : DecidableEq α
    f : Finsupp (Prod α β) M
    ⊢ HasSubset.Subset f.curry.support (f.support.biUnion fun a => Singleton.singl …
  -/
  refine Finset.Subset.trans support_sum ?_
  /-
    α : Type u_1
    β : Type u_2
    M : Type u_5
    inst✝¹ : AddCommMonoid M
    inst✝ : DecidableEq α
    f : Finsupp (Prod α β) M
    ⊢ HasSubset.Subset (f.support.biUnion fun a => (Finsupp.single a.1 (Finsupp.si …
  -/
  exact Finset.biUnion_mono fun a _ => support_single_subset
  /-
    🎉 no goals
  -/


/-- `Finsupp.sumElim f g` maps `inl x` to `f x` and `inr y` to `g y`. -/
@[simps support]
def sumElim {α β γ : Type*} [Zero γ] (f : α →₀ γ) (g : β →₀ γ) : α ⊕ β →₀ γ where
  support := f.support.disjSum g.support
  toFun := Sum.elim f g
                          /-
                            α✝ : Type u_1
                            β✝ : Type u_2
                            γ✝ : Type u_3
                            ι : Type u_4
                            M : Type u_5
                            M' : Type u_6
                            N : Type u_7
                            P : Type u_8
                            G : Type u_9
                            H : Type u_10
                            R : Type u_11
                            S : Type u_12
                            α : Type u_13
                            β : Type u_14
                            γ : Type u_15
                            inst✝ : Zero γ
                            f : Finsupp α γ
                            g : Finsupp β γ
                            ⊢ ∀ (a : Sum α β), Iff (Membership.mem (f.support.disjSum g.support) a) (Ne (S …
                          -/
  mem_support_toFun := by simp
                          /-
                            🎉 no goals
                          -/


@[simp, norm_cast]
theorem coe_sumElim {α β γ : Type*} [Zero γ] (f : α →₀ γ) (g : β →₀ γ) :
    ⇑(sumElim f g) = Sum.elim f g :=
  rfl


theorem sumElim_apply {α β γ : Type*} [Zero γ] (f : α →₀ γ) (g : β →₀ γ) (x : α ⊕ β) :
    sumElim f g x = Sum.elim f g x :=
  rfl


theorem sumElim_inl {α β γ : Type*} [Zero γ] (f : α →₀ γ) (g : β →₀ γ) (x : α) :
    sumElim f g (Sum.inl x) = f x :=
  rfl


theorem sumElim_inr {α β γ : Type*} [Zero γ] (f : α →₀ γ) (g : β →₀ γ) (x : β) :
    sumElim f g (Sum.inr x) = g x :=
  rfl


@[to_additive]
lemma prod_sumElim {ι₁ ι₂ α M : Type*} [Zero α] [CommMonoid M]
    (f₁ : ι₁ →₀ α) (f₂ : ι₂ →₀ α) (g : ι₁ ⊕ ι₂ → α → M) :
    (f₁.sumElim f₂).prod g = f₁.prod (g ∘ Sum.inl) * f₂.prod (g ∘ Sum.inr) := by
  /-
    ι₁ : Type u_13
    ι₂ : Type u_14
    α : Type u_15
    M : Type u_16
    inst✝¹ : Zero α
    inst✝ : CommMonoid M
    f₁ : Finsupp ι₁ α
    f₂ : Finsupp ι₂ α
    g : Sum ι₁ ι₂ → α → M
    ⊢ Eq ((f₁.sumElim f₂).prod g) (HMul.hMul (f₁.prod (Function.comp g Sum.inl)) ( …
  -/
  simp [Finsupp.prod, Finset.prod_disj_sum]
  /-
    🎉 no goals
  -/


/-- The equivalence between `(α ⊕ β) →₀ γ` and `(α →₀ γ) × (β →₀ γ)`.

This is the `Finsupp` version of `Equiv.sum_arrow_equiv_prod_arrow`. -/
@[simps apply symm_apply]
def sumFinsuppEquivProdFinsupp {α β γ : Type*} [Zero γ] : (α ⊕ β →₀ γ) ≃ (α →₀ γ) × (β →₀ γ) where
  toFun f :=
    ⟨f.comapDomain Sum.inl Sum.inl_injective.injOn,
      f.comapDomain Sum.inr Sum.inr_injective.injOn⟩
  invFun fg := sumElim fg.1 fg.2
  left_inv f := by
    /-
      α✝ : Type u_1
      β✝ : Type u_2
      γ✝ : Type u_3
      ι : Type u_4
      M : Type u_5
      M' : Type u_6
      N : Type u_7
      P : Type u_8
      G : Type u_9
      H : Type u_10
      R : Type u_11
      S : Type u_12
      α : Type u_13
      β : Type u_14
      γ : Type u_15
      inst✝ : Zero γ
      f : Finsupp (Sum α β) γ
      ⊢ Eq ((fun fg => fg.1.sumElim fg.2) ((fun f => { fst := Finsupp.comapDomain Su …
    -/
    ext ab
    /-
      case h
      α✝ : Type u_1
      β✝ : Type u_2
      γ✝ : Type u_3
      ι : Type u_4
      M : Type u_5
      M' : Type u_6
      N : Type u_7
      P : Type u_8
      G : Type u_9
      H : Type u_10
      R : Type u_11
      S : Type u_12
      α : Type u_13
      β : Type u_14
      γ : Type u_15
      inst✝ : Zero γ
      f : Finsupp (Sum α β) γ
      ab : Sum α β
      ⊢ Eq (((fun fg => fg.1.sumElim fg.2) ((fun f => { fst := Finsupp.comapDomain S …
    -/
                           /-
                             🎉 no goals
                           -/
    cases' ab with a b <;> simp
                           /-
                             🎉 no goals
                           -/
                     /-
                       α✝ : Type u_1
                       β✝ : Type u_2
                       γ✝ : Type u_3
                       ι : Type u_4
                       M : Type u_5
                       M' : Type u_6
                       N : Type u_7
                       P : Type u_8
                       G : Type u_9
                       H : Type u_10
                       R : Type u_11
                       S : Type u_12
                       α : Type u_13
                       β : Type u_14
                       γ : Type u_15
                       inst✝ : Zero γ
                       fg : Prod (Finsupp α γ) (Finsupp β γ)
                       ⊢ Eq ((fun f => { fst := Finsupp.comapDomain Sum.inl f ⋯, snd := Finsupp.comap …
                     -/
                             /-
                               🎉 no goals
                             -/
  right_inv fg := by ext <;> simp
                             /-
                               🎉 no goals
                             -/


theorem fst_sumFinsuppEquivProdFinsupp {α β γ : Type*} [Zero γ] (f : α ⊕ β →₀ γ) (x : α) :
    (sumFinsuppEquivProdFinsupp f).1 x = f (Sum.inl x) :=
  rfl


theorem snd_sumFinsuppEquivProdFinsupp {α β γ : Type*} [Zero γ] (f : α ⊕ β →₀ γ) (y : β) :
    (sumFinsuppEquivProdFinsupp f).2 y = f (Sum.inr y) :=
  rfl


theorem sumFinsuppEquivProdFinsupp_symm_inl {α β γ : Type*} [Zero γ] (fg : (α →₀ γ) × (β →₀ γ))
    (x : α) : (sumFinsuppEquivProdFinsupp.symm fg) (Sum.inl x) = fg.1 x :=
  rfl


theorem sumFinsuppEquivProdFinsupp_symm_inr {α β γ : Type*} [Zero γ] (fg : (α →₀ γ) × (β →₀ γ))
    (y : β) : (sumFinsuppEquivProdFinsupp.symm fg) (Sum.inr y) = fg.2 y :=
  rfl


/-- The additive equivalence between `(α ⊕ β) →₀ M` and `(α →₀ M) × (β →₀ M)`.

This is the `Finsupp` version of `Equiv.sum_arrow_equiv_prod_arrow`. -/
@[simps! apply symm_apply]
def sumFinsuppAddEquivProdFinsupp {α β : Type*} : (α ⊕ β →₀ M) ≃+ (α →₀ M) × (β →₀ M) :=
  { sumFinsuppEquivProdFinsupp with
    map_add' := by
      /-
        α✝ : Type u_1
        β✝ : Type u_2
        γ : Type u_3
        ι : Type u_4
        M : Type u_5
        M' : Type u_6
        N : Type u_7
        P : Type u_8
        G : Type u_9
        H : Type u_10
        R : Type u_11
        S : Type u_12
        inst✝ : AddMonoid M
        α : Type u_13
        β : Type u_14
        ⊢ ∀ (x y : Finsupp (Sum α β) M), Eq (__src✝.toFun (HAdd.hAdd x y)) (HAdd.hAdd  …
      -/
      intros
      /-
        α✝ : Type u_1
        β✝ : Type u_2
        γ : Type u_3
        ι : Type u_4
        M : Type u_5
        M' : Type u_6
        N : Type u_7
        P : Type u_8
        G : Type u_9
        H : Type u_10
        R : Type u_11
        S : Type u_12
        inst✝ : AddMonoid M
        α : Type u_13
        β : Type u_14
        x✝ y✝ : Finsupp (Sum α β) M
        ⊢ Eq (__src✝.toFun (HAdd.hAdd x✝ y✝)) (HAdd.hAdd (__src✝.toFun x✝) (__src✝.toF …
      -/
      ext <;>
        simp only [Equiv.toFun_as_coe, Prod.fst_add, Prod.snd_add, add_apply,
          snd_sumFinsuppEquivProdFinsupp, fst_sumFinsuppEquivProdFinsupp] }


theorem fst_sumFinsuppAddEquivProdFinsupp {α β : Type*} (f : α ⊕ β →₀ M) (x : α) :
    (sumFinsuppAddEquivProdFinsupp f).1 x = f (Sum.inl x) :=
  rfl


theorem snd_sumFinsuppAddEquivProdFinsupp {α β : Type*} (f : α ⊕ β →₀ M) (y : β) :
    (sumFinsuppAddEquivProdFinsupp f).2 y = f (Sum.inr y) :=
  rfl


theorem sumFinsuppAddEquivProdFinsupp_symm_inl {α β : Type*} (fg : (α →₀ M) × (β →₀ M)) (x : α) :
    (sumFinsuppAddEquivProdFinsupp.symm fg) (Sum.inl x) = fg.1 x :=
  rfl


theorem sumFinsuppAddEquivProdFinsupp_symm_inr {α β : Type*} (fg : (α →₀ M) × (β →₀ M)) (y : β) :
    (sumFinsuppAddEquivProdFinsupp.symm fg) (Sum.inr y) = fg.2 y :=
  rfl


@[simp, nolint simpNF] -- `simpNF` incorrectly complains the LHS doesn't simplify.
theorem single_smul (a b : α) (f : α → M) (r : R) : single a r b • f a = single a (r • f b) b := by
  /-
    α : Type u_1
    M : Type u_5
    R : Type u_11
    inst✝² : Zero M
    inst✝¹ : MonoidWithZero R
    inst✝ : MulActionWithZero R M
    a b : α
    f : α → M
    r : R
    ⊢ Eq (HSMul.hSMul ((Finsupp.single a r) b) (f a)) ((Finsupp.single a (HSMul.hS …
  -/
                         /-
                           🎉 no goals
                         -/
  by_cases h : a = b <;> simp [h]
                         /-
                           🎉 no goals
                         -/


/-- Scalar multiplication acting on the domain.

This is not an instance as it would conflict with the action on the range.
See the `instance_diamonds` test for examples of such conflicts. -/
def comapSMul : SMul G (α →₀ M) where smul g := mapDomain (g • ·)


theorem comapSMul_def (g : G) (f : α →₀ M) : g • f = mapDomain (g • ·) f :=
  rfl


@[simp]
theorem comapSMul_single (g : G) (a : α) (b : M) : g • single a b = single (g • a) b :=
  mapDomain_single


/-- `Finsupp.comapSMul` is multiplicative -/
def comapMulAction : MulAction G (α →₀ M) where
                   /-
                     α : Type u_1
                     β : Type u_2
                     γ : Type u_3
                     ι : Type u_4
                     M : Type u_5
                     M' : Type u_6
                     N : Type u_7
                     P : Type u_8
                     G : Type u_9
                     H : Type u_10
                     R : Type u_11
                     S : Type u_12
                     inst✝² : Monoid G
                     inst✝¹ : MulAction G α
                     inst✝ : AddCommMonoid M
                     f : Finsupp α M
                     ⊢ Eq (HSMul.hSMul 1 f) f
                   -/
  one_smul f := by rw [comapSMul_def, one_smul_eq_id, mapDomain_id]
                   /-
                     🎉 no goals
                   -/
  mul_smul g g' f := by
    /-
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      ι : Type u_4
      M : Type u_5
      M' : Type u_6
      N : Type u_7
      P : Type u_8
      G : Type u_9
      H : Type u_10
      R : Type u_11
      S : Type u_12
      inst✝² : Monoid G
      inst✝¹ : MulAction G α
      inst✝ : AddCommMonoid M
      g g' : G
      f : Finsupp α M
      ⊢ Eq (HSMul.hSMul (HMul.hMul g g') f) (HSMul.hSMul g (HSMul.hSMul g' f))
    -/
    rw [comapSMul_def, comapSMul_def, comapSMul_def, ← comp_smul_left, mapDomain_comp]
    /-
      🎉 no goals
    -/


/-- `Finsupp.comapSMul` is distributive -/
def comapDistribMulAction : DistribMulAction G (α →₀ M) where
  smul_zero g := by
    /-
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      ι : Type u_4
      M : Type u_5
      M' : Type u_6
      N : Type u_7
      P : Type u_8
      G : Type u_9
      H : Type u_10
      R : Type u_11
      S : Type u_12
      inst✝² : Monoid G
      inst✝¹ : MulAction G α
      inst✝ : AddCommMonoid M
      g : G
      ⊢ Eq (HSMul.hSMul g 0) 0
    -/
    ext a
    /-
      case h
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      ι : Type u_4
      M : Type u_5
      M' : Type u_6
      N : Type u_7
      P : Type u_8
      G : Type u_9
      H : Type u_10
      R : Type u_11
      S : Type u_12
      inst✝² : Monoid G
      inst✝¹ : MulAction G α
      inst✝ : AddCommMonoid M
      g : G
      a : α
      ⊢ Eq ((HSMul.hSMul g 0) a) (0 a)
    -/
    simp only [comapSMul_def]
    /-
      case h
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      ι : Type u_4
      M : Type u_5
      M' : Type u_6
      N : Type u_7
      P : Type u_8
      G : Type u_9
      H : Type u_10
      R : Type u_11
      S : Type u_12
      inst✝² : Monoid G
      inst✝¹ : MulAction G α
      inst✝ : AddCommMonoid M
      g : G
      a : α
      ⊢ Eq ((Finsupp.mapDomain (fun x => HSMul.hSMul g x) 0) a) (0 a)
    -/
    simp
    /-
      🎉 no goals
    -/
  smul_add g f f' := by
    /-
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      ι : Type u_4
      M : Type u_5
      M' : Type u_6
      N : Type u_7
      P : Type u_8
      G : Type u_9
      H : Type u_10
      R : Type u_11
      S : Type u_12
      inst✝² : Monoid G
      inst✝¹ : MulAction G α
      inst✝ : AddCommMonoid M
      g : G
      f f' : Finsupp α M
      ⊢ Eq (HSMul.hSMul g (HAdd.hAdd f f')) (HAdd.hAdd (HSMul.hSMul g f) (HSMul.hSMu …
    -/
    ext
    /-
      case h
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      ι : Type u_4
      M : Type u_5
      M' : Type u_6
      N : Type u_7
      P : Type u_8
      G : Type u_9
      H : Type u_10
      R : Type u_11
      S : Type u_12
      inst✝² : Monoid G
      inst✝¹ : MulAction G α
      inst✝ : AddCommMonoid M
      g : G
      f f' : Finsupp α M
      a✝ : α
      ⊢ Eq ((HSMul.hSMul g (HAdd.hAdd f f')) a✝) ((HAdd.hAdd (HSMul.hSMul g f) (HSMu …
    -/
    simp only [comapSMul_def]
    /-
      case h
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      ι : Type u_4
      M : Type u_5
      M' : Type u_6
      N : Type u_7
      P : Type u_8
      G : Type u_9
      H : Type u_10
      R : Type u_11
      S : Type u_12
      inst✝² : Monoid G
      inst✝¹ : MulAction G α
      inst✝ : AddCommMonoid M
      g : G
      f f' : Finsupp α M
      a✝ : α
      ⊢ Eq ((Finsupp.mapDomain (fun x => HSMul.hSMul g x) (HAdd.hAdd f f')) a✝) ((HA …
    -/
    simp [mapDomain_add]
    /-
      🎉 no goals
    -/


/-- When `G` is a group, `Finsupp.comapSMul` acts by precomposition with the action of `g⁻¹`.
-/
@[simp]
theorem comapSMul_apply (g : G) (f : α →₀ M) (a : α) : (g • f) a = f (g⁻¹ • a) := by
  /-
    α : Type u_1
    M : Type u_5
    G : Type u_9
    inst✝² : Group G
    inst✝¹ : MulAction G α
    inst✝ : AddCommMonoid M
    g : G
    f : Finsupp α M
    a : α
    ⊢ Eq ((HSMul.hSMul g f) a) (f (HSMul.hSMul (Inv.inv g) a))
  -/
  conv_lhs => rw [← smul_inv_smul g a]
  /-
    α : Type u_1
    M : Type u_5
    G : Type u_9
    inst✝² : Group G
    inst✝¹ : MulAction G α
    inst✝ : AddCommMonoid M
    g : G
    f : Finsupp α M
    a : α
    ⊢ Eq ((HSMul.hSMul g f) (HSMul.hSMul g (HSMul.hSMul (Inv.inv g) a))) (f (HSMul …
  -/
  exact mapDomain_apply (MulAction.injective g) _ (g⁻¹ • a)
  /-
    🎉 no goals
  -/


theorem _root_.IsSMulRegular.finsupp [Zero M] [SMulZeroClass R M] {k : R}
    (hk : IsSMulRegular M k) : IsSMulRegular (α →₀ M) k :=
  fun _ _ h => ext fun i => hk (DFunLike.congr_fun h i)


instance faithfulSMul [Nonempty α] [Zero M] [SMulZeroClass R M] [FaithfulSMul R M] :
    FaithfulSMul R (α →₀ M) where
  eq_of_smul_eq_smul h :=
    let ⟨a⟩ := ‹Nonempty α›
                                       /-
                                         α : Type u_1
                                         β : Type u_2
                                         γ : Type u_3
                                         ι : Type u_4
                                         M : Type u_5
                                         M' : Type u_6
                                         N : Type u_7
                                         P : Type u_8
                                         G : Type u_9
                                         H : Type u_10
                                         R : Type u_11
                                         S : Type u_12
                                         inst✝³ : Nonempty α
                                         inst✝² : Zero M
                                         inst✝¹ : SMulZeroClass R M
                                         inst✝ : FaithfulSMul R M
                                         m₁✝ m₂✝ : R
                                         h : ∀ (a : Finsupp α M), Eq (HSMul.hSMul m₁✝ a) (HSMul.hSMul m₂✝ a)
                                         a : α
                                         m : M
                                         ⊢ Eq (HSMul.hSMul m₁✝ m) (HSMul.hSMul m₂✝ m)
                                       -/
    eq_of_smul_eq_smul fun m : M => by simpa using DFunLike.congr_fun (h (single a m)) a
                                       /-
                                         🎉 no goals
                                       -/


instance distribMulAction [Monoid R] [AddMonoid M] [DistribMulAction R M] :
    DistribMulAction R (α →₀ M) :=
  { Finsupp.distribSMul _ _ with
    one_smul := fun x => ext fun y => one_smul R (x y)
    mul_smul := fun r s x => ext fun y => mul_smul r s (x y) }


instance module [Semiring R] [AddCommMonoid M] [Module R M] : Module R (α →₀ M) :=
  { toDistribMulAction := Finsupp.distribMulAction α M
    zero_smul := fun _ => ext fun _ => zero_smul _ _
    add_smul := fun _ _ _ => ext fun _ => add_smul _ _ _ }


@[simp]
theorem support_smul_eq [Semiring R] [AddCommMonoid M] [Module R M] [NoZeroSMulDivisors R M] {b : R}
    (hb : b ≠ 0) {g : α →₀ M} : (b • g).support = g.support :=
                         /-
                           α : Type u_1
                           M : Type u_5
                           R : Type u_11
                           inst✝³ : Semiring R
                           inst✝² : AddCommMonoid M
                           inst✝¹ : Module R M
                           inst✝ : NoZeroSMulDivisors R M
                           b : R
                           hb : Ne b 0
                           g : Finsupp α M
                           a : α
                           ⊢ Iff (Membership.mem (HSMul.hSMul b g).support a) (Membership.mem g.support a)
                         -/
  Finset.ext fun a => by simp [Finsupp.smul_apply, hb]
                         /-
                           🎉 no goals
                         -/


@[simp]
theorem filter_smul {_ : Monoid R} [AddMonoid M] [DistribMulAction R M] {b : R} {v : α →₀ M} :
    (b • v).filter p = b • v.filter p :=
  DFunLike.coe_injective <| by
    /-
      α : Type u_1
      M : Type u_5
      R : Type u_11
      p : α → Prop
      inst✝² : DecidablePred p
      x✝ : Monoid R
      inst✝¹ : AddMonoid M
      inst✝ : DistribMulAction R M
      b : R
      v : Finsupp α M
      ⊢ Eq ((fun f => ⇑f) (Finsupp.filter p (HSMul.hSMul b v))) ((fun f => ⇑f) (HSMu …
    -/
    simp only [filter_eq_indicator, coe_smul]
    /-
      α : Type u_1
      M : Type u_5
      R : Type u_11
      p : α → Prop
      inst✝² : DecidablePred p
      x✝ : Monoid R
      inst✝¹ : AddMonoid M
      inst✝ : DistribMulAction R M
      b : R
      v : Finsupp α M
      ⊢ Eq ((setOf fun x => p x).indicator (HSMul.hSMul b ⇑v)) (HSMul.hSMul b ((setO …
    -/
    exact Set.indicator_const_smul { x | p x } b v
    /-
      🎉 no goals
    -/


theorem mapDomain_smul {_ : Monoid R} [AddCommMonoid M] [DistribMulAction R M] {f : α → β} (b : R)
    (v : α →₀ M) : mapDomain f (b • v) = b • mapDomain f v :=
  mapDomain_mapRange _ _ _ _ (smul_add b)

-- Porting note: removed `simp` because `simpNF` can prove it.

theorem smul_single' {_ : Semiring R} (c : R) (a : α) (b : R) :
    c • Finsupp.single a b = Finsupp.single a (c * b) :=
  smul_single _ _ _


theorem smul_single_one [Semiring R] (a : α) (b : R) : b • single a (1 : R) = single a b := by
  /-
    α : Type u_1
    R : Type u_11
    inst✝ : Semiring R
    a : α
    b : R
    ⊢ Eq (HSMul.hSMul b (Finsupp.single a 1)) (Finsupp.single a b)
  -/
  rw [smul_single, smul_eq_mul, mul_one]
  /-
    🎉 no goals
  -/


theorem comapDomain_smul [AddMonoid M] [Monoid R] [DistribMulAction R M] {f : α → β} (r : R)
    (v : β →₀ M) (hfv : Set.InjOn f (f ⁻¹' ↑v.support))
    (hfrv : Set.InjOn f (f ⁻¹' ↑(r • v).support) :=
      hfv.mono <| Set.preimage_mono <| Finset.coe_subset.mpr support_smul) :
    comapDomain f (r • v) hfrv = r • comapDomain f v hfv := by
  /-
    α : Type u_1
    β : Type u_2
    M : Type u_5
    R : Type u_11
    inst✝² : AddMonoid M
    inst✝¹ : Monoid R
    inst✝ : DistribMulAction R M
    f : α → β
    r : R
    v : Finsupp β M
    hfv : Set.InjOn f (Set.preimage f ↑v.support)
    hfrv : optParam (Set.InjOn f (Set.preimage f ↑(HSMul.hSMul r v).support)) ⋯
    ⊢ Eq (Finsupp.comapDomain f (HSMul.hSMul r v) hfrv) (HSMul.hSMul r (Finsupp.co …
  -/
  ext
  /-
    case h
    α : Type u_1
    β : Type u_2
    M : Type u_5
    R : Type u_11
    inst✝² : AddMonoid M
    inst✝¹ : Monoid R
    inst✝ : DistribMulAction R M
    f : α → β
    r : R
    v : Finsupp β M
    hfv : Set.InjOn f (Set.preimage f ↑v.support)
    hfrv : optParam (Set.InjOn f (Set.preimage f ↑(HSMul.hSMul r v).support)) ⋯
    a✝ : α
    ⊢ Eq ((Finsupp.comapDomain f (HSMul.hSMul r v) hfrv) a✝) ((HSMul.hSMul r (Fins …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- A version of `Finsupp.comapDomain_smul` that's easier to use. -/
theorem comapDomain_smul_of_injective [AddMonoid M] [Monoid R] [DistribMulAction R M] {f : α → β}
    (hf : Function.Injective f) (r : R) (v : β →₀ M) :
    comapDomain f (r • v) hf.injOn = r • comapDomain f v hf.injOn :=
  comapDomain_smul _ _ _ _


theorem sum_smul_index [Semiring R] [AddCommMonoid M] {g : α →₀ R} {b : R} {h : α → R → M}
    (h0 : ∀ i, h i 0 = 0) : (b • g).sum h = g.sum fun i a => h i (b * a) :=
  Finsupp.sum_mapRange_index h0


theorem sum_smul_index' [AddMonoid M] [DistribSMul R M] [AddCommMonoid N] {g : α →₀ M} {b : R}
    {h : α → M → N} (h0 : ∀ i, h i 0 = 0) : (b • g).sum h = g.sum fun i c => h i (b • c) :=
  Finsupp.sum_mapRange_index h0


/-- A version of `Finsupp.sum_smul_index'` for bundled additive maps. -/
theorem sum_smul_index_addMonoidHom [AddMonoid M] [AddCommMonoid N] [DistribSMul R M] {g : α →₀ M}
    {b : R} {h : α → M →+ N} : ((b • g).sum fun a => h a) = g.sum fun i c => h i (b • c) :=
  sum_mapRange_index fun i => (h i).map_zero


instance noZeroSMulDivisors [Zero R] [Zero M] [SMulZeroClass R M] {ι : Type*}
    [NoZeroSMulDivisors R M] : NoZeroSMulDivisors R (ι →₀ M) :=
  ⟨fun h => or_iff_not_imp_left.mpr fun hc => Finsupp.ext fun i =>
    (eq_zero_or_eq_zero_of_smul_eq_zero (DFunLike.ext_iff.mp h i)).resolve_left hc⟩


/-- `Finsupp.single` as a `DistribMulActionSemiHom`.

See also `Finsupp.lsingle` for the version as a linear map. -/
def DistribMulActionHom.single (a : α) : M →+[R] α →₀ M :=
  { singleAddHom a with
    map_smul' := fun k m => by
      /-
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        ι : Type u_4
        M : Type u_5
        M' : Type u_6
        N : Type u_7
        P : Type u_8
        G : Type u_9
        H : Type u_10
        R : Type u_11
        S : Type u_12
        inst✝⁴ : Monoid R
        inst✝³ : AddMonoid M
        inst✝² : AddMonoid N
        inst✝¹ : DistribMulAction R M
        inst✝ : DistribMulAction R N
        a : α
        k : R
        m : M
        ⊢ Eq ((↑__src✝).toFun (HSMul.hSMul k m)) (HSMul.hSMul ((MonoidHom.id R) k) ((↑ …
      -/
      simp only
      /-
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        ι : Type u_4
        M : Type u_5
        M' : Type u_6
        N : Type u_7
        P : Type u_8
        G : Type u_9
        H : Type u_10
        R : Type u_11
        S : Type u_12
        inst✝⁴ : Monoid R
        inst✝³ : AddMonoid M
        inst✝² : AddMonoid N
        inst✝¹ : DistribMulAction R M
        inst✝ : DistribMulAction R N
        a : α
        k : R
        m : M
        ⊢ Eq ((↑(Finsupp.singleAddHom a)).toFun (HSMul.hSMul k m)) (HSMul.hSMul ((Mono …
      -/
      show singleAddHom a (k • m) = k • singleAddHom a m
      /-
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        ι : Type u_4
        M : Type u_5
        M' : Type u_6
        N : Type u_7
        P : Type u_8
        G : Type u_9
        H : Type u_10
        R : Type u_11
        S : Type u_12
        inst✝⁴ : Monoid R
        inst✝³ : AddMonoid M
        inst✝² : AddMonoid N
        inst✝¹ : DistribMulAction R M
        inst✝ : DistribMulAction R N
        a : α
        k : R
        m : M
        ⊢ Eq ((Finsupp.singleAddHom a) (HSMul.hSMul k m)) (HSMul.hSMul k ((Finsupp.sin …
      -/
      change Finsupp.single a (k • m) = k • (Finsupp.single a m)
      -- Porting note: because `singleAddHom_apply` is missing
      /-
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        ι : Type u_4
        M : Type u_5
        M' : Type u_6
        N : Type u_7
        P : Type u_8
        G : Type u_9
        H : Type u_10
        R : Type u_11
        S : Type u_12
        inst✝⁴ : Monoid R
        inst✝³ : AddMonoid M
        inst✝² : AddMonoid N
        inst✝¹ : DistribMulAction R M
        inst✝ : DistribMulAction R N
        a : α
        k : R
        m : M
        ⊢ Eq (Finsupp.single a (HSMul.hSMul k m)) (HSMul.hSMul k (Finsupp.single a m))
      -/
      simp only [smul_single] }
      /-
        🎉 no goals
      -/


theorem distribMulActionHom_ext {f g : (α →₀ M) →+[R] N}
    (h : ∀ (a : α) (m : M), f (single a m) = g (single a m)) : f = g :=
  DistribMulActionHom.toAddMonoidHom_injective <| addHom_ext h


/-- See note [partially-applied ext lemmas]. -/
@[ext]
theorem distribMulActionHom_ext' {f g : (α →₀ M) →+[R] N}
    (h : ∀ a : α, f.comp (DistribMulActionHom.single a) = g.comp (DistribMulActionHom.single a)) :
    f = g :=
  distribMulActionHom_ext fun a => DistribMulActionHom.congr_fun (h a)


/-- The `Finsupp` version of `Pi.unique`. -/
instance uniqueOfRight [Subsingleton R] : Unique (α →₀ R) :=
  DFunLike.coe_injective.unique


/-- The `Finsupp` version of `Pi.uniqueOfIsEmpty`. -/
instance uniqueOfLeft [IsEmpty α] : Unique (α →₀ R) :=
  DFunLike.coe_injective.unique


/-- Combine finitely supported functions over `{a // P a}` and `{a // ¬P a}`, by case-splitting on
`P a`. -/
@[simps]
def piecewise (f : Subtype P →₀ M) (g : {a // ¬ P a} →₀ M) : α →₀ M where
  toFun a := if h : P a then f ⟨a, h⟩ else g ⟨a, h⟩
  support := (f.support.map (.subtype _)).disjUnion (g.support.map (.subtype _)) <| by
    simp_rw [Finset.disjoint_left, mem_map, forall_exists_index, Embedding.coe_subtype,
      Subtype.forall, Subtype.exists]
    /-
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      ι : Type u_4
      M✝ : Type u_5
      M' : Type u_6
      N : Type u_7
      P✝ : Type u_8
      G : Type u_9
      H : Type u_10
      R : Type u_11
      S : Type u_12
      M : Type u_13
      inst✝¹ : Zero M
      P : α → Prop
      inst✝ : DecidablePred P
      f : Finsupp (Subtype P) M
      g : Finsupp (Subtype fun a => Not (P a)) M
      ⊢ ∀ ⦃a : α⦄ (a_1 : α) (b : P a_1), And (Membership.mem f.support ⟨a_1, b⟩) (Eq …
    -/
    rintro _ a ha ⟨-, rfl⟩ ⟨b, hb, -, rfl⟩
    /-
      case intro.intro.intro.intro
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      ι : Type u_4
      M✝ : Type u_5
      M' : Type u_6
      N : Type u_7
      P✝ : Type u_8
      G : Type u_9
      H : Type u_10
      R : Type u_11
      S : Type u_12
      M : Type u_13
      inst✝¹ : Zero M
      P : α → Prop
      inst✝ : DecidablePred P
      f : Finsupp (Subtype P) M
      g : Finsupp (Subtype fun a => Not (P a)) M
      b : α
      hb : Not (P b)
      ha : P b
      ⊢ False
    -/
    exact hb ha
    /-
      🎉 no goals
    -/
  mem_support_toFun a := by
    /-
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      ι : Type u_4
      M✝ : Type u_5
      M' : Type u_6
      N : Type u_7
      P✝ : Type u_8
      G : Type u_9
      H : Type u_10
      R : Type u_11
      S : Type u_12
      M : Type u_13
      inst✝¹ : Zero M
      P : α → Prop
      inst✝ : DecidablePred P
      f : Finsupp (Subtype P) M
      g : Finsupp (Subtype fun a => Not (P a)) M
      a : α
      ⊢ Iff (Membership.mem ((Finset.map (Function.Embedding.subtype P) f.support).d …
    -/
                          /-
                            🎉 no goals
                          -/
    by_cases ha : P a <;> simp [ha]
                          /-
                            🎉 no goals
                          -/


@[simp]
theorem subtypeDomain_piecewise (f : Subtype P →₀ M) (g : {a // ¬ P a} →₀ M) :
    subtypeDomain P (f.piecewise g) = f :=
  Finsupp.ext fun a => dif_pos a.prop


@[simp]
theorem subtypeDomain_not_piecewise (f : Subtype P →₀ M) (g : {a // ¬ P a} →₀ M) :
    subtypeDomain (¬P ·) (f.piecewise g) = g :=
  Finsupp.ext fun a => dif_neg a.prop


/-- Extend the domain of a `Finsupp` by using `0` where `P x` does not hold. -/
@[simps! support toFun]
def extendDomain (f : Subtype P →₀ M) : α →₀ M := piecewise f 0


theorem extendDomain_eq_embDomain_subtype (f : Subtype P →₀ M) :
    extendDomain f = embDomain (.subtype _) f := by
  /-
    α : Type u_1
    M : Type u_13
    inst✝¹ : Zero M
    P : α → Prop
    inst✝ : DecidablePred P
    f : Finsupp (Subtype P) M
    ⊢ Eq f.extendDomain (Finsupp.embDomain (Function.Embedding.subtype P) f)
  -/
  ext a
  /-
    case h
    α : Type u_1
    M : Type u_13
    inst✝¹ : Zero M
    P : α → Prop
    inst✝ : DecidablePred P
    f : Finsupp (Subtype P) M
    a : α
    ⊢ Eq (f.extendDomain a) ((Finsupp.embDomain (Function.Embedding.subtype P) f) a)
  -/
  by_cases h : P a
    /-
      case pos
      α : Type u_1
      M : Type u_13
      inst✝¹ : Zero M
      P : α → Prop
      inst✝ : DecidablePred P
      f : Finsupp (Subtype P) M
      a : α
      h : P a
      ⊢ Eq (f.extendDomain a) ((Finsupp.embDomain (Function.Embedding.subtype P) f) a)
    -/
  · refine Eq.trans ?_ (embDomain_apply (.subtype P) f (Subtype.mk a h)).symm
    /-
      case pos
      α : Type u_1
      M : Type u_13
      inst✝¹ : Zero M
      P : α → Prop
      inst✝ : DecidablePred P
      f : Finsupp (Subtype P) M
      a : α
      h : P a
      ⊢ Eq (f.extendDomain a) (f ⟨a, h⟩)
    -/
    simp [h]
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      M : Type u_13
      inst✝¹ : Zero M
      P : α → Prop
      inst✝ : DecidablePred P
      f : Finsupp (Subtype P) M
      a : α
      h : Not (P a)
      ⊢ Eq (f.extendDomain a) ((Finsupp.embDomain (Function.Embedding.subtype P) f) a)
    -/
  · rw [embDomain_notin_range, extendDomain_toFun, dif_neg h]
    /-
      case neg.h
      α : Type u_1
      M : Type u_13
      inst✝¹ : Zero M
      P : α → Prop
      inst✝ : DecidablePred P
      f : Finsupp (Subtype P) M
      a : α
      h : Not (P a)
      ⊢ Not (Membership.mem (Set.range ⇑(Function.Embedding.subtype P)) a)
    -/
    simp [h]
    /-
      🎉 no goals
    -/


theorem support_extendDomain_subset (f : Subtype P →₀ M) :
    ↑(f.extendDomain).support ⊆ {x | P x} := by
  /-
    α : Type u_1
    M : Type u_13
    inst✝¹ : Zero M
    P : α → Prop
    inst✝ : DecidablePred P
    f : Finsupp (Subtype P) M
    ⊢ HasSubset.Subset (↑f.extendDomain.support) (setOf fun x => P x)
  -/
  intro x
  /-
    α : Type u_1
    M : Type u_13
    inst✝¹ : Zero M
    P : α → Prop
    inst✝ : DecidablePred P
    f : Finsupp (Subtype P) M
    x : α
    ⊢ Membership.mem (↑f.extendDomain.support) x → Membership.mem (setOf fun x =>  …
  -/
  rw [extendDomain_support, mem_coe, mem_map, Embedding.coe_subtype]
  /-
    α : Type u_1
    M : Type u_13
    inst✝¹ : Zero M
    P : α → Prop
    inst✝ : DecidablePred P
    f : Finsupp (Subtype P) M
    x : α
    ⊢ (Exists fun a => And (Membership.mem f.support a) (Eq (↑a) x)) → Membership. …
  -/
  rintro ⟨x, -, rfl⟩
  /-
    case intro.intro
    α : Type u_1
    M : Type u_13
    inst✝¹ : Zero M
    P : α → Prop
    inst✝ : DecidablePred P
    f : Finsupp (Subtype P) M
    x : Subtype P
    ⊢ Membership.mem (setOf fun x => P x) ↑x
  -/
  exact x.prop
  /-
    🎉 no goals
  -/


@[simp]
theorem subtypeDomain_extendDomain (f : Subtype P →₀ M) :
    subtypeDomain P f.extendDomain = f :=
  subtypeDomain_piecewise _ _


theorem extendDomain_subtypeDomain (f : α →₀ M) (hf : ∀ a ∈ f.support, P a) :
    (subtypeDomain P f).extendDomain = f := by
  /-
    α : Type u_1
    M : Type u_13
    inst✝¹ : Zero M
    P : α → Prop
    inst✝ : DecidablePred P
    f : Finsupp α M
    hf : ∀ (a : α), Membership.mem f.support a → P a
    ⊢ Eq (Finsupp.subtypeDomain P f).extendDomain f
  -/
  ext a
  /-
    case h
    α : Type u_1
    M : Type u_13
    inst✝¹ : Zero M
    P : α → Prop
    inst✝ : DecidablePred P
    f : Finsupp α M
    hf : ∀ (a : α), Membership.mem f.support a → P a
    a : α
    ⊢ Eq ((Finsupp.subtypeDomain P f).extendDomain a) (f a)
  -/
  by_cases h : P a
    /-
      case pos
      α : Type u_1
      M : Type u_13
      inst✝¹ : Zero M
      P : α → Prop
      inst✝ : DecidablePred P
      f : Finsupp α M
      hf : ∀ (a : α), Membership.mem f.support a → P a
      a : α
      h : P a
      ⊢ Eq ((Finsupp.subtypeDomain P f).extendDomain a) (f a)
    -/
  · exact dif_pos h
    /-
      🎉 no goals
    -/
  · #adaptation_note
    /-- Prior to nightly-2024-06-18, this `rw` was done by `dsimp`. -/
    /-
      case neg
      α : Type u_1
      M : Type u_13
      inst✝¹ : Zero M
      P : α → Prop
      inst✝ : DecidablePred P
      f : Finsupp α M
      hf : ∀ (a : α), Membership.mem f.support a → P a
      a : α
      h : Not (P a)
      ⊢ Eq ((Finsupp.subtypeDomain P f).extendDomain a) (f a)
    -/
    rw [extendDomain_toFun]
    /-
      case neg
      α : Type u_1
      M : Type u_13
      inst✝¹ : Zero M
      P : α → Prop
      inst✝ : DecidablePred P
      f : Finsupp α M
      hf : ∀ (a : α), Membership.mem f.support a → P a
      a : α
      h : Not (P a)
      ⊢ Eq (dite (P a) (fun h => (Finsupp.subtypeDomain P f) ⟨a, h⟩) fun h => 0) (f a)
    -/
    dsimp
    /-
      case neg
      α : Type u_1
      M : Type u_13
      inst✝¹ : Zero M
      P : α → Prop
      inst✝ : DecidablePred P
      f : Finsupp α M
      hf : ∀ (a : α), Membership.mem f.support a → P a
      a : α
      h : Not (P a)
      ⊢ Eq (ite (P a) (f a) 0) (f a)
    -/
    rw [if_neg h, eq_comm, ← not_mem_support_iff]
    /-
      case neg
      α : Type u_1
      M : Type u_13
      inst✝¹ : Zero M
      P : α → Prop
      inst✝ : DecidablePred P
      f : Finsupp α M
      hf : ∀ (a : α), Membership.mem f.support a → P a
      a : α
      h : Not (P a)
      ⊢ Not (Membership.mem f.support a)
    -/
    refine mt ?_ h
    /-
      case neg
      α : Type u_1
      M : Type u_13
      inst✝¹ : Zero M
      P : α → Prop
      inst✝ : DecidablePred P
      f : Finsupp α M
      hf : ∀ (a : α), Membership.mem f.support a → P a
      a : α
      h : Not (P a)
      ⊢ Membership.mem f.support a → P a
    -/
    exact @hf _
    /-
      🎉 no goals
    -/


@[simp]
theorem extendDomain_single (a : Subtype P) (m : M) :
    (single a m).extendDomain = single a.val m := by
  /-
    α : Type u_1
    M : Type u_13
    inst✝¹ : Zero M
    P : α → Prop
    inst✝ : DecidablePred P
    a : Subtype P
    m : M
    ⊢ Eq (Finsupp.single a m).extendDomain (Finsupp.single (↑a) m)
  -/
  ext a'
  #adaptation_note
  /-- Prior to nightly-2024-06-18, this `rw` was instead `dsimp only`. -/
  /-
    case h
    α : Type u_1
    M : Type u_13
    inst✝¹ : Zero M
    P : α → Prop
    inst✝ : DecidablePred P
    a : Subtype P
    m : M
    a' : α
    ⊢ Eq ((Finsupp.single a m).extendDomain a') ((Finsupp.single (↑a) m) a')
  -/
  rw [extendDomain_toFun]
  /-
    case h
    α : Type u_1
    M : Type u_13
    inst✝¹ : Zero M
    P : α → Prop
    inst✝ : DecidablePred P
    a : Subtype P
    m : M
    a' : α
    ⊢ Eq (dite (P a') (fun h => (Finsupp.single a m) ⟨a', h⟩) fun h => 0) ((Finsup …
  -/
  obtain rfl | ha := eq_or_ne a.val a'
    /-
      case h.inl
      α : Type u_1
      M : Type u_13
      inst✝¹ : Zero M
      P : α → Prop
      inst✝ : DecidablePred P
      a : Subtype P
      m : M
      ⊢ Eq (dite (P ↑a) (fun h => (Finsupp.single a m) ⟨↑a, h⟩) fun h => 0) ((Finsup …
    -/
  · simp_rw [single_eq_same, dif_pos a.prop]
    /-
      🎉 no goals
    -/
    /-
      case h.inr
      α : Type u_1
      M : Type u_13
      inst✝¹ : Zero M
      P : α → Prop
      inst✝ : DecidablePred P
      a : Subtype P
      m : M
      a' : α
      ha : Ne (↑a) a'
      ⊢ Eq (dite (P a') (fun h => (Finsupp.single a m) ⟨a', h⟩) fun h => 0) ((Finsup …
    -/
  · simp_rw [single_eq_of_ne ha, dite_eq_right_iff]
    /-
      case h.inr
      α : Type u_1
      M : Type u_13
      inst✝¹ : Zero M
      P : α → Prop
      inst✝ : DecidablePred P
      a : Subtype P
      m : M
      a' : α
      ha : Ne (↑a) a'
      ⊢ ∀ (h : P a'), Eq ((Finsupp.single a m) ⟨a', h⟩) 0
    -/
    intro h
    /-
      case h.inr
      α : Type u_1
      M : Type u_13
      inst✝¹ : Zero M
      P : α → Prop
      inst✝ : DecidablePred P
      a : Subtype P
      m : M
      a' : α
      ha : Ne (↑a) a'
      h : P a'
      ⊢ Eq ((Finsupp.single a m) ⟨a', h⟩) 0
    -/
    rw [single_eq_of_ne]
    /-
      case h.inr
      α : Type u_1
      M : Type u_13
      inst✝¹ : Zero M
      P : α → Prop
      inst✝ : DecidablePred P
      a : Subtype P
      m : M
      a' : α
      ha : Ne (↑a) a'
      h : P a'
      ⊢ Ne a ⟨a', h⟩
    -/
    simp [Subtype.ext_iff, ha]
    /-
      🎉 no goals
    -/


/-- Given an `AddCommMonoid M` and `s : Set α`, `restrictSupportEquiv s M` is the `Equiv`
between the subtype of finitely supported functions with support contained in `s` and
the type of finitely supported functions from `s`. -/
def restrictSupportEquiv (s : Set α) (M : Type*) [AddCommMonoid M] :
    { f : α →₀ M // ↑f.support ⊆ s } ≃ (s →₀ M) where
  toFun f := subtypeDomain (· ∈ s) f.1
  invFun f := letI := Classical.decPred (· ∈ s); ⟨f.extendDomain, support_extendDomain_subset _⟩
  left_inv f :=
    letI := Classical.decPred (· ∈ s); Subtype.ext <| extendDomain_subtypeDomain f.1 f.prop
  right_inv _ := letI := Classical.decPred (· ∈ s); subtypeDomain_extendDomain _


/-- Given `AddCommMonoid M` and `e : α ≃ β`, `domCongr e` is the corresponding `Equiv` between
`α →₀ M` and `β →₀ M`.

This is `Finsupp.equivCongrLeft` as an `AddEquiv`. -/
@[simps apply]
protected def domCongr [AddCommMonoid M] (e : α ≃ β) : (α →₀ M) ≃+ (β →₀ M) where
  toFun := equivMapDomain e
  invFun := equivMapDomain e.symm
  left_inv v := by
    /-
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      ι : Type u_4
      M : Type u_5
      M' : Type u_6
      N : Type u_7
      P : Type u_8
      G : Type u_9
      H : Type u_10
      R : Type u_11
      S : Type u_12
      inst✝ : AddCommMonoid M
      e : Equiv α β
      v : Finsupp α M
      ⊢ Eq (Finsupp.equivMapDomain e.symm (Finsupp.equivMapDomain e v)) v
    -/
    simp only [← equivMapDomain_trans, Equiv.self_trans_symm]
    /-
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      ι : Type u_4
      M : Type u_5
      M' : Type u_6
      N : Type u_7
      P : Type u_8
      G : Type u_9
      H : Type u_10
      R : Type u_11
      S : Type u_12
      inst✝ : AddCommMonoid M
      e : Equiv α β
      v : Finsupp α M
      ⊢ Eq (Finsupp.equivMapDomain (Equiv.refl α) v) v
    -/
    exact equivMapDomain_refl _
    /-
      🎉 no goals
    -/
  right_inv := by
    /-
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      ι : Type u_4
      M : Type u_5
      M' : Type u_6
      N : Type u_7
      P : Type u_8
      G : Type u_9
      H : Type u_10
      R : Type u_11
      S : Type u_12
      inst✝ : AddCommMonoid M
      e : Equiv α β
      ⊢ Function.RightInverse (Finsupp.equivMapDomain e.symm) (Finsupp.equivMapDomai …
    -/
    intro v
    /-
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      ι : Type u_4
      M : Type u_5
      M' : Type u_6
      N : Type u_7
      P : Type u_8
      G : Type u_9
      H : Type u_10
      R : Type u_11
      S : Type u_12
      inst✝ : AddCommMonoid M
      e : Equiv α β
      v : Finsupp β M
      ⊢ Eq (Finsupp.equivMapDomain e (Finsupp.equivMapDomain e.symm v)) v
    -/
    simp only [← equivMapDomain_trans, Equiv.symm_trans_self]
    /-
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      ι : Type u_4
      M : Type u_5
      M' : Type u_6
      N : Type u_7
      P : Type u_8
      G : Type u_9
      H : Type u_10
      R : Type u_11
      S : Type u_12
      inst✝ : AddCommMonoid M
      e : Equiv α β
      v : Finsupp β M
      ⊢ Eq (Finsupp.equivMapDomain (Equiv.refl β) v) v
    -/
    exact equivMapDomain_refl _
    /-
      🎉 no goals
    -/
                     /-
                       α : Type u_1
                       β : Type u_2
                       γ : Type u_3
                       ι : Type u_4
                       M : Type u_5
                       M' : Type u_6
                       N : Type u_7
                       P : Type u_8
                       G : Type u_9
                       H : Type u_10
                       R : Type u_11
                       S : Type u_12
                       inst✝ : AddCommMonoid M
                       e : Equiv α β
                       a b : Finsupp α M
                       ⊢ Eq ({ toFun := Finsupp.equivMapDomain e, invFun := Finsupp.equivMapDomain e. …
                     -/
  map_add' a b := by simp only [equivMapDomain_eq_mapDomain]; exact mapDomain_add
                                                              /-
                                                                🎉 no goals
                                                              -/


@[simp]
theorem domCongr_refl [AddCommMonoid M] :
    Finsupp.domCongr (Equiv.refl α) = AddEquiv.refl (α →₀ M) :=
  AddEquiv.ext fun _ => equivMapDomain_refl _


@[simp]
theorem domCongr_symm [AddCommMonoid M] (e : α ≃ β) :
    (Finsupp.domCongr e).symm = (Finsupp.domCongr e.symm : (β →₀ M) ≃+ (α →₀ M)) :=
  AddEquiv.ext fun _ => rfl


@[simp]
theorem domCongr_trans [AddCommMonoid M] (e : α ≃ β) (f : β ≃ γ) :
    (Finsupp.domCongr e).trans (Finsupp.domCongr f) =
      (Finsupp.domCongr (e.trans f) : (α →₀ M) ≃+ _) :=
  AddEquiv.ext fun _ => (equivMapDomain_trans _ _ _).symm


/-- Given `l`, a finitely supported function from the sigma type `Σ (i : ι), αs i` to `M` and
an index element `i : ι`, `split l i` is the `i`th component of `l`,
a finitely supported function from `as i` to `M`.

This is the `Finsupp` version of `Sigma.curry`.
-/
def split (i : ι) : αs i →₀ M :=
  l.comapDomain (Sigma.mk i) fun _ _ _ _ hx => heq_iff_eq.1 (Sigma.mk.inj_iff.mp hx).2
  -- Porting note: it seems like Lean 4 never generated the `Sigma.mk.inj` lemma?


theorem split_apply (i : ι) (x : αs i) : split l i x = l ⟨i, x⟩ := by
  /-
    ι : Type u_4
    M : Type u_5
    αs : ι → Type u_13
    inst✝ : Zero M
    l : Finsupp (Sigma fun i => αs i) M
    i : ι
    x : αs i
    ⊢ Eq ((l.split i) x) (l ⟨i, x⟩)
  -/
  dsimp only [split]
  /-
    ι : Type u_4
    M : Type u_5
    αs : ι → Type u_13
    inst✝ : Zero M
    l : Finsupp (Sigma fun i => αs i) M
    i : ι
    x : αs i
    ⊢ Eq ((Finsupp.comapDomain (Sigma.mk i) l ⋯) x) (l ⟨i, x⟩)
  -/
  rw [comapDomain_apply]
  /-
    🎉 no goals
  -/


/-- Given `l`, a finitely supported function from the sigma type `Σ (i : ι), αs i` to `β`,
`split_support l` is the finset of indices in `ι` that appear in the support of `l`. -/
def splitSupport (l : (Σi, αs i) →₀ M) : Finset ι :=
  haveI := Classical.decEq ι
  l.support.image Sigma.fst


theorem mem_splitSupport_iff_nonzero (i : ι) : i ∈ splitSupport l ↔ split l i ≠ 0 := by
  rw [splitSupport, @mem_image _ _ (Classical.decEq _), Ne, ← support_eq_empty, ← Ne, ←
    Finset.nonempty_iff_ne_empty, split, comapDomain, Finset.Nonempty]
  -- Porting note (https://github.com/leanprover-community/mathlib4/issues/10754): had to add the `Classical.decEq` instance manually
  simp only [exists_prop, Finset.mem_preimage, exists_and_right, exists_eq_right, mem_support_iff,
    Sigma.exists, Ne]


/-- Given `l`, a finitely supported function from the sigma type `Σ i, αs i` to `β` and
an `ι`-indexed family `g` of functions from `(αs i →₀ β)` to `γ`, `split_comp` defines a
finitely supported function from the index type `ι` to `γ` given by composing `g i` with
`split l i`. -/
def splitComp [Zero N] (g : ∀ i, (αs i →₀ M) → N) (hg : ∀ i x, x = 0 ↔ g i x = 0) : ι →₀ N where
  support := splitSupport l
  toFun i := g i (split l i)
  mem_support_toFun := by
    /-
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      ι : Type u_4
      M : Type u_5
      M' : Type u_6
      N : Type u_7
      P : Type u_8
      G : Type u_9
      H : Type u_10
      R : Type u_11
      S : Type u_12
      αs : ι → Type u_13
      inst✝¹ : Zero M
      l : Finsupp (Sigma fun i => αs i) M
      inst✝ : Zero N
      g : (i : ι) → Finsupp (αs i) M → N
      hg : ∀ (i : ι) (x : Finsupp (αs i) M), Iff (Eq x 0) (Eq (g i x) 0)
      ⊢ ∀ (a : ι), Iff (Membership.mem l.splitSupport a) (Ne ((fun i => g i (l.split …
    -/
    intro i
    /-
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      ι : Type u_4
      M : Type u_5
      M' : Type u_6
      N : Type u_7
      P : Type u_8
      G : Type u_9
      H : Type u_10
      R : Type u_11
      S : Type u_12
      αs : ι → Type u_13
      inst✝¹ : Zero M
      l : Finsupp (Sigma fun i => αs i) M
      inst✝ : Zero N
      g : (i : ι) → Finsupp (αs i) M → N
      hg : ∀ (i : ι) (x : Finsupp (αs i) M), Iff (Eq x 0) (Eq (g i x) 0)
      i : ι
      ⊢ Iff (Membership.mem l.splitSupport i) (Ne ((fun i => g i (l.split i)) i) 0)
    -/
    rw [mem_splitSupport_iff_nonzero, not_iff_not, hg]
    /-
      🎉 no goals
    -/


theorem sigma_support : l.support = l.splitSupport.sigma fun i => (l.split i).support := by
  simp only [Finset.ext_iff, splitSupport, split, comapDomain, @mem_image _ _ (Classical.decEq _),
    mem_preimage, Sigma.forall, mem_sigma]
  -- Porting note (https://github.com/leanprover-community/mathlib4/issues/10754): had to add the `Classical.decEq` instance manually
  /-
    ι : Type u_4
    M : Type u_5
    αs : ι → Type u_13
    inst✝ : Zero M
    l : Finsupp (Sigma fun i => αs i) M
    ⊢ ∀ (a : ι) (b : αs a), Iff (Membership.mem l.support ⟨a, b⟩) (And (Exists fun …
  -/
  tauto
  /-
    🎉 no goals
  -/


theorem sigma_sum [AddCommMonoid N] (f : (Σi : ι, αs i) → M → N) :
    l.sum f = ∑ i ∈ splitSupport l, (split l i).sum fun (a : αs i) b => f ⟨i, a⟩ b := by
  /-
    ι : Type u_4
    M : Type u_5
    N : Type u_7
    αs : ι → Type u_13
    inst✝¹ : Zero M
    l : Finsupp (Sigma fun i => αs i) M
    inst✝ : AddCommMonoid N
    f : (Sigma fun i => αs i) → M → N
    ⊢ Eq (l.sum f) (l.splitSupport.sum fun i => (l.split i).sum fun a b => f ⟨i, a …
  -/
  simp only [sum, sigma_support, sum_sigma, split_apply]
  /-
    🎉 no goals
  -/


/-- On a `Fintype η`, `Finsupp.split` is an equivalence between `(Σ (j : η), ιs j) →₀ α`
and `Π j, (ιs j →₀ α)`.

This is the `Finsupp` version of `Equiv.Pi_curry`. -/
noncomputable def sigmaFinsuppEquivPiFinsupp : ((Σj, ιs j) →₀ α) ≃ ∀ j, ιs j →₀ α where
  toFun := split
  invFun f :=
    onFinset (Finset.univ.sigma fun j => (f j).support) (fun ji => f ji.1 ji.2) fun _ hg =>
      Finset.mem_sigma.mpr ⟨Finset.mem_univ _, mem_support_iff.mpr hg⟩
  left_inv f := by
    /-
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      ι : Type u_4
      M : Type u_5
      M' : Type u_6
      N : Type u_7
      P : Type u_8
      G : Type u_9
      H : Type u_10
      R : Type u_11
      S : Type u_12
      αs : ι → Type u_13
      inst✝² : Zero M
      l : Finsupp (Sigma fun i => αs i) M
      η : Type u_14
      inst✝¹ : Fintype η
      ιs : η → Type u_15
      inst✝ : Zero α
      f : Finsupp (Sigma fun j => ιs j) α
      ⊢ Eq ((fun f => Finsupp.onFinset (Finset.univ.sigma fun j => (f j).support) (f …
    -/
    ext
    /-
      case h
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      ι : Type u_4
      M : Type u_5
      M' : Type u_6
      N : Type u_7
      P : Type u_8
      G : Type u_9
      H : Type u_10
      R : Type u_11
      S : Type u_12
      αs : ι → Type u_13
      inst✝² : Zero M
      l : Finsupp (Sigma fun i => αs i) M
      η : Type u_14
      inst✝¹ : Fintype η
      ιs : η → Type u_15
      inst✝ : Zero α
      f : Finsupp (Sigma fun j => ιs j) α
      a✝ : Sigma fun j => ιs j
      ⊢ Eq (((fun f => Finsupp.onFinset (Finset.univ.sigma fun j => (f j).support) ( …
    -/
    simp [split]
    /-
      🎉 no goals
    -/
  right_inv f := by
    /-
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      ι : Type u_4
      M : Type u_5
      M' : Type u_6
      N : Type u_7
      P : Type u_8
      G : Type u_9
      H : Type u_10
      R : Type u_11
      S : Type u_12
      αs : ι → Type u_13
      inst✝² : Zero M
      l : Finsupp (Sigma fun i => αs i) M
      η : Type u_14
      inst✝¹ : Fintype η
      ιs : η → Type u_15
      inst✝ : Zero α
      f : (j : η) → Finsupp (ιs j) α
      ⊢ Eq ((fun f => Finsupp.onFinset (Finset.univ.sigma fun j => (f j).support) (f …
    -/
    ext
    /-
      case h.h
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      ι : Type u_4
      M : Type u_5
      M' : Type u_6
      N : Type u_7
      P : Type u_8
      G : Type u_9
      H : Type u_10
      R : Type u_11
      S : Type u_12
      αs : ι → Type u_13
      inst✝² : Zero M
      l : Finsupp (Sigma fun i => αs i) M
      η : Type u_14
      inst✝¹ : Fintype η
      ιs : η → Type u_15
      inst✝ : Zero α
      f : (j : η) → Finsupp (ιs j) α
      x✝ : η
      a✝ : ιs x✝
      ⊢ Eq ((((fun f => Finsupp.onFinset (Finset.univ.sigma fun j => (f j).support)  …
    -/
    simp [split]
    /-
      🎉 no goals
    -/


@[simp]
theorem sigmaFinsuppEquivPiFinsupp_apply (f : (Σj, ιs j) →₀ α) (j i) :
    sigmaFinsuppEquivPiFinsupp f j i = f ⟨j, i⟩ :=
  rfl


/-- On a `Fintype η`, `Finsupp.split` is an additive equivalence between
`(Σ (j : η), ιs j) →₀ α` and `Π j, (ιs j →₀ α)`.

This is the `AddEquiv` version of `Finsupp.sigmaFinsuppEquivPiFinsupp`.
-/
noncomputable def sigmaFinsuppAddEquivPiFinsupp {α : Type*} {ιs : η → Type*} [AddMonoid α] :
    ((Σj, ιs j) →₀ α) ≃+ ∀ j, ιs j →₀ α :=
  { sigmaFinsuppEquivPiFinsupp with
    map_add' := fun f g => by
      /-
        α✝ : Type u_1
        β : Type u_2
        γ : Type u_3
        ι : Type u_4
        M : Type u_5
        M' : Type u_6
        N : Type u_7
        P : Type u_8
        G : Type u_9
        H : Type u_10
        R : Type u_11
        S : Type u_12
        αs : ι → Type u_13
        inst✝³ : Zero M
        l : Finsupp (Sigma fun i => αs i) M
        η : Type u_14
        inst✝² : Fintype η
        ιs✝ : η → Type u_15
        inst✝¹ : Zero α✝
        α : Type u_16
        ιs : η → Type u_17
        inst✝ : AddMonoid α
        f g : Finsupp (Sigma fun j => ιs j) α
        ⊢ Eq (__src✝.toFun (HAdd.hAdd f g)) (HAdd.hAdd (__src✝.toFun f) (__src✝.toFun  …
      -/
      ext
      /-
        case h.h
        α✝ : Type u_1
        β : Type u_2
        γ : Type u_3
        ι : Type u_4
        M : Type u_5
        M' : Type u_6
        N : Type u_7
        P : Type u_8
        G : Type u_9
        H : Type u_10
        R : Type u_11
        S : Type u_12
        αs : ι → Type u_13
        inst✝³ : Zero M
        l : Finsupp (Sigma fun i => αs i) M
        η : Type u_14
        inst✝² : Fintype η
        ιs✝ : η → Type u_15
        inst✝¹ : Zero α✝
        α : Type u_16
        ιs : η → Type u_17
        inst✝ : AddMonoid α
        f g : Finsupp (Sigma fun j => ιs j) α
        x✝ : η
        a✝ : ιs x✝
        ⊢ Eq ((__src✝.toFun (HAdd.hAdd f g) x✝) a✝) ((HAdd.hAdd (__src✝.toFun f) (__sr …
      -/
      simp }
      /-
        🎉 no goals
      -/


@[simp]
theorem sigmaFinsuppAddEquivPiFinsupp_apply {α : Type*} {ιs : η → Type*} [AddMonoid α]
    (f : (Σj, ιs j) →₀ α) (j i) : sigmaFinsuppAddEquivPiFinsupp f j i = f ⟨j, i⟩ :=
  rfl


