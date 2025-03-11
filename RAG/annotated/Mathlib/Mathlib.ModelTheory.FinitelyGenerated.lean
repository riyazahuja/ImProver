/-- A substructure of `M` is finitely generated if it is the closure of a finite subset of `M`. -/
def FG (N : L.Substructure M) : Prop :=
  ∃ S : Finset M, closure L S = N


theorem fg_def {N : L.Substructure M} : N.FG ↔ ∃ S : Set M, S.Finite ∧ closure L S = N :=
  ⟨fun ⟨t, h⟩ => ⟨_, Finset.finite_toSet t, h⟩, by
    /-
      L : FirstOrder.Language
      M : Type u_1
      inst✝ : L.Structure M
      N : L.Substructure M
      ⊢ (Exists fun S => And S.Finite (Eq ((FirstOrder.Language.Substructure.closure …
    -/
    rintro ⟨t', h, rfl⟩
    /-
      case intro.intro
      L : FirstOrder.Language
      M : Type u_1
      inst✝ : L.Structure M
      t' : Set M
      h : t'.Finite
      ⊢ ((FirstOrder.Language.Substructure.closure L).toFun t').FG
    -/
    rcases Finite.exists_finset_coe h with ⟨t, rfl⟩
    /-
      case intro.intro.intro
      L : FirstOrder.Language
      M : Type u_1
      inst✝ : L.Structure M
      t : Finset M
      h : (↑t).Finite
      ⊢ ((FirstOrder.Language.Substructure.closure L).toFun ↑t).FG
    -/
    exact ⟨t, rfl⟩⟩
    /-
      🎉 no goals
    -/


theorem fg_iff_exists_fin_generating_family {N : L.Substructure M} :
    N.FG ↔ ∃ (n : ℕ) (s : Fin n → M), closure L (range s) = N := by
  /-
    L : FirstOrder.Language
    M : Type u_1
    inst✝ : L.Structure M
    N : L.Substructure M
    ⊢ Iff N.FG (Exists fun n => Exists fun s => Eq ((FirstOrder.Language.Substruct …
  -/
  rw [fg_def]
  /-
    L : FirstOrder.Language
    M : Type u_1
    inst✝ : L.Structure M
    N : L.Substructure M
    ⊢ Iff (Exists fun S => And S.Finite (Eq ((FirstOrder.Language.Substructure.clo …
  -/
  constructor
    /-
      case mp
      L : FirstOrder.Language
      M : Type u_1
      inst✝ : L.Structure M
      N : L.Substructure M
      ⊢ (Exists fun S => And S.Finite (Eq ((FirstOrder.Language.Substructure.closure …
    -/
  · rintro ⟨S, Sfin, hS⟩
    /-
      case mp.intro.intro
      L : FirstOrder.Language
      M : Type u_1
      inst✝ : L.Structure M
      N : L.Substructure M
      S : Set M
      Sfin : S.Finite
      hS : Eq ((FirstOrder.Language.Substructure.closure L).toFun S) N
      ⊢ Exists fun n => Exists fun s => Eq ((FirstOrder.Language.Substructure.closur …
    -/
    obtain ⟨n, f, rfl⟩ := Sfin.fin_embedding
    /-
      case mp.intro.intro.intro.intro
      L : FirstOrder.Language
      M : Type u_1
      inst✝ : L.Structure M
      N : L.Substructure M
      n : Nat
      f : Function.Embedding (Fin n) M
      Sfin : (Set.range ⇑f).Finite
      hS : Eq ((FirstOrder.Language.Substructure.closure L).toFun (Set.range ⇑f)) N
      ⊢ Exists fun n => Exists fun s => Eq ((FirstOrder.Language.Substructure.closur …
    -/
    exact ⟨n, f, hS⟩
    /-
      🎉 no goals
    -/
    /-
      case mpr
      L : FirstOrder.Language
      M : Type u_1
      inst✝ : L.Structure M
      N : L.Substructure M
      ⊢ (Exists fun n => Exists fun s => Eq ((FirstOrder.Language.Substructure.closu …
    -/
  · rintro ⟨n, s, hs⟩
    /-
      case mpr.intro.intro
      L : FirstOrder.Language
      M : Type u_1
      inst✝ : L.Structure M
      N : L.Substructure M
      n : Nat
      s : Fin n → M
      hs : Eq ((FirstOrder.Language.Substructure.closure L).toFun (Set.range s)) N
      ⊢ Exists fun S => And S.Finite (Eq ((FirstOrder.Language.Substructure.closure  …
    -/
    exact ⟨range s, finite_range s, hs⟩
    /-
      🎉 no goals
    -/


theorem fg_bot : (⊥ : L.Substructure M).FG :=
         /-
           L : FirstOrder.Language
           M : Type u_1
           inst✝ : L.Structure M
           ⊢ Eq ((FirstOrder.Language.Substructure.closure L).toFun ↑EmptyCollection.empt …
         -/
  ⟨∅, by rw [Finset.coe_empty, closure_empty]⟩
         /-
           🎉 no goals
         -/


instance instInhabited_fg : Inhabited { S : L.Substructure M // S.FG } := ⟨⊥, fg_bot⟩


theorem fg_closure {s : Set M} (hs : s.Finite) : FG (closure L s) :=
                   /-
                     L : FirstOrder.Language
                     M : Type u_1
                     inst✝ : L.Structure M
                     s : Set M
                     hs : s.Finite
                     ⊢ Eq ((FirstOrder.Language.Substructure.closure L).toFun ↑hs.toFinset) ((First …
                   -/
  ⟨hs.toFinset, by rw [hs.coe_toFinset]⟩
                   /-
                     🎉 no goals
                   -/


theorem fg_closure_singleton (x : M) : FG (closure L ({x} : Set M)) :=
  fg_closure (finite_singleton x)


theorem FG.sup {N₁ N₂ : L.Substructure M} (hN₁ : N₁.FG) (hN₂ : N₂.FG) : (N₁ ⊔ N₂).FG :=
  let ⟨t₁, ht₁⟩ := fg_def.1 hN₁
  let ⟨t₂, ht₂⟩ := fg_def.1 hN₂
                                           /-
                                             L : FirstOrder.Language
                                             M : Type u_1
                                             inst✝ : L.Structure M
                                             N₁ N₂ : L.Substructure M
                                             hN₁ : N₁.FG
                                             hN₂ : N₂.FG
                                             t₁ : Set M
                                             ht₁ : And t₁.Finite (Eq ((FirstOrder.Language.Substructure.closure L).toFun t₁ …
                                             t₂ : Set M
                                             ht₂ : And t₂.Finite (Eq ((FirstOrder.Language.Substructure.closure L).toFun t₂ …
                                             ⊢ Eq ((FirstOrder.Language.Substructure.closure L).toFun (Union.union t₁ t₂))  …
                                           -/
  fg_def.2 ⟨t₁ ∪ t₂, ht₁.1.union ht₂.1, by rw [closure_union, ht₁.2, ht₂.2]⟩
                                           /-
                                             🎉 no goals
                                           -/


theorem FG.map {N : Type*} [L.Structure N] (f : M →[L] N) {s : L.Substructure M} (hs : s.FG) :
    (s.map f).FG :=
  let ⟨t, ht⟩ := fg_def.1 hs
                                     /-
                                       L : FirstOrder.Language
                                       M : Type u_1
                                       inst✝¹ : L.Structure M
                                       N : Type u_2
                                       inst✝ : L.Structure N
                                       f : L.Hom M N
                                       s : L.Substructure M
                                       hs : s.FG
                                       t : Set M
                                       ht : And t.Finite (Eq ((FirstOrder.Language.Substructure.closure L).toFun t) s)
                                       ⊢ Eq ((FirstOrder.Language.Substructure.closure L).toFun (Set.image (⇑f) t)) ( …
                                     -/
  fg_def.2 ⟨f '' t, ht.1.image _, by rw [closure_image, ht.2]⟩
                                     /-
                                       🎉 no goals
                                     -/


theorem FG.of_map_embedding {N : Type*} [L.Structure N] (f : M ↪[L] N) {s : L.Substructure M}
    (hs : (s.map f.toHom).FG) : s.FG := by
  /-
    L : FirstOrder.Language
    M : Type u_1
    inst✝¹ : L.Structure M
    N : Type u_2
    inst✝ : L.Structure N
    f : L.Embedding M N
    s : L.Substructure M
    hs : (FirstOrder.Language.Substructure.map f.toHom s).FG
    ⊢ s.FG
  -/
  rcases hs with ⟨t, h⟩
  /-
    case intro
    L : FirstOrder.Language
    M : Type u_1
    inst✝¹ : L.Structure M
    N : Type u_2
    inst✝ : L.Structure N
    f : L.Embedding M N
    s : L.Substructure M
    t : Finset N
    h : Eq ((FirstOrder.Language.Substructure.closure L).toFun ↑t) (FirstOrder.Lan …
    ⊢ s.FG
  -/
  rw [fg_def]
  /-
    case intro
    L : FirstOrder.Language
    M : Type u_1
    inst✝¹ : L.Structure M
    N : Type u_2
    inst✝ : L.Structure N
    f : L.Embedding M N
    s : L.Substructure M
    t : Finset N
    h : Eq ((FirstOrder.Language.Substructure.closure L).toFun ↑t) (FirstOrder.Lan …
    ⊢ Exists fun S => And S.Finite (Eq ((FirstOrder.Language.Substructure.closure  …
  -/
  refine ⟨f ⁻¹' t, t.finite_toSet.preimage f.injective.injOn, ?_⟩
  /-
    case intro
    L : FirstOrder.Language
    M : Type u_1
    inst✝¹ : L.Structure M
    N : Type u_2
    inst✝ : L.Structure N
    f : L.Embedding M N
    s : L.Substructure M
    t : Finset N
    h : Eq ((FirstOrder.Language.Substructure.closure L).toFun ↑t) (FirstOrder.Lan …
    ⊢ Eq ((FirstOrder.Language.Substructure.closure L).toFun (Set.preimage ⇑f ↑t)) s
  -/
  have hf : Function.Injective f.toHom := f.injective
  /-
    case intro
    L : FirstOrder.Language
    M : Type u_1
    inst✝¹ : L.Structure M
    N : Type u_2
    inst✝ : L.Structure N
    f : L.Embedding M N
    s : L.Substructure M
    t : Finset N
    h : Eq ((FirstOrder.Language.Substructure.closure L).toFun ↑t) (FirstOrder.Lan …
    hf : Function.Injective ⇑f.toHom
    ⊢ Eq ((FirstOrder.Language.Substructure.closure L).toFun (Set.preimage ⇑f ↑t)) s
  -/
  refine map_injective_of_injective hf ?_
  /-
    case intro
    L : FirstOrder.Language
    M : Type u_1
    inst✝¹ : L.Structure M
    N : Type u_2
    inst✝ : L.Structure N
    f : L.Embedding M N
    s : L.Substructure M
    t : Finset N
    h : Eq ((FirstOrder.Language.Substructure.closure L).toFun ↑t) (FirstOrder.Lan …
    hf : Function.Injective ⇑f.toHom
    ⊢ Eq (FirstOrder.Language.Substructure.map f.toHom ((FirstOrder.Language.Subst …
  -/
  rw [← h, map_closure, Embedding.coe_toHom, image_preimage_eq_of_subset]
  /-
    case intro
    L : FirstOrder.Language
    M : Type u_1
    inst✝¹ : L.Structure M
    N : Type u_2
    inst✝ : L.Structure N
    f : L.Embedding M N
    s : L.Substructure M
    t : Finset N
    h : Eq ((FirstOrder.Language.Substructure.closure L).toFun ↑t) (FirstOrder.Lan …
    hf : Function.Injective ⇑f.toHom
    ⊢ HasSubset.Subset (↑t) (Set.range ⇑f)
  -/
  intro x hx
  /-
    case intro
    L : FirstOrder.Language
    M : Type u_1
    inst✝¹ : L.Structure M
    N : Type u_2
    inst✝ : L.Structure N
    f : L.Embedding M N
    s : L.Substructure M
    t : Finset N
    h : Eq ((FirstOrder.Language.Substructure.closure L).toFun ↑t) (FirstOrder.Lan …
    hf : Function.Injective ⇑f.toHom
    x : N
    hx : Membership.mem (↑t) x
    ⊢ Membership.mem (Set.range ⇑f) x
  -/
  have h' := subset_closure (L := L) hx
  /-
    case intro
    L : FirstOrder.Language
    M : Type u_1
    inst✝¹ : L.Structure M
    N : Type u_2
    inst✝ : L.Structure N
    f : L.Embedding M N
    s : L.Substructure M
    t : Finset N
    h : Eq ((FirstOrder.Language.Substructure.closure L).toFun ↑t) (FirstOrder.Lan …
    hf : Function.Injective ⇑f.toHom
    x : N
    hx : Membership.mem (↑t) x
    h' : Membership.mem (↑((FirstOrder.Language.Substructure.closure L).toFun ↑t)) x
    ⊢ Membership.mem (Set.range ⇑f) x
  -/
  rw [h] at h'
  /-
    case intro
    L : FirstOrder.Language
    M : Type u_1
    inst✝¹ : L.Structure M
    N : Type u_2
    inst✝ : L.Structure N
    f : L.Embedding M N
    s : L.Substructure M
    t : Finset N
    h : Eq ((FirstOrder.Language.Substructure.closure L).toFun ↑t) (FirstOrder.Lan …
    hf : Function.Injective ⇑f.toHom
    x : N
    hx : Membership.mem (↑t) x
    h' : Membership.mem (↑(FirstOrder.Language.Substructure.map f.toHom s)) x
    ⊢ Membership.mem (Set.range ⇑f) x
  -/
  exact Hom.map_le_range h'
  /-
    🎉 no goals
  -/


theorem FG.of_finite {s : L.Substructure M} [h : Finite s] : s.FG :=
                             /-
                               L : FirstOrder.Language
                               M : Type u_1
                               inst✝ : L.Structure M
                               s : L.Substructure M
                               h : Finite (Subtype fun x => Membership.mem s x)
                               ⊢ Eq ((FirstOrder.Language.Substructure.closure L).toFun ↑(Set.Finite.toFinset …
                             -/
  ⟨Set.Finite.toFinset h, by simp only [Finite.coe_toFinset, closure_eq]⟩
                             /-
                               🎉 no goals
                             -/


theorem FG.finite [L.IsRelational] {S : L.Substructure M} (h : S.FG) : Finite S := by
  /-
    L : FirstOrder.Language
    M : Type u_1
    inst✝¹ : L.Structure M
    inst✝ : L.IsRelational
    S : L.Substructure M
    h : S.FG
    ⊢ Finite (Subtype fun x => Membership.mem S x)
  -/
  obtain ⟨s, rfl⟩ := h
  /-
    case intro
    L : FirstOrder.Language
    M : Type u_1
    inst✝¹ : L.Structure M
    inst✝ : L.IsRelational
    s : Finset M
    ⊢ Finite (Subtype fun x => Membership.mem ((FirstOrder.Language.Substructure.c …
  -/
  have hs := s.finite_toSet
  /-
    case intro
    L : FirstOrder.Language
    M : Type u_1
    inst✝¹ : L.Structure M
    inst✝ : L.IsRelational
    s : Finset M
    hs : (↑s).Finite
    ⊢ Finite (Subtype fun x => Membership.mem ((FirstOrder.Language.Substructure.c …
  -/
  rw [← closure_eq_of_isRelational L (s : Set M)] at hs
  /-
    case intro
    L : FirstOrder.Language
    M : Type u_1
    inst✝¹ : L.Structure M
    inst✝ : L.IsRelational
    s : Finset M
    hs : (↑((FirstOrder.Language.Substructure.closure L).toFun ↑s)).Finite
    ⊢ Finite (Subtype fun x => Membership.mem ((FirstOrder.Language.Substructure.c …
  -/
  exact hs
  /-
    🎉 no goals
  -/


theorem fg_iff_finite [L.IsRelational] {S : L.Substructure M} : S.FG ↔ Finite S :=
  ⟨FG.finite, fun _ => FG.of_finite⟩


/-- A substructure of `M` is countably generated if it is the closure of a countable subset of `M`.
-/
def CG (N : L.Substructure M) : Prop :=
  ∃ S : Set M, S.Countable ∧ closure L S = N


theorem cg_def {N : L.Substructure M} : N.CG ↔ ∃ S : Set M, S.Countable ∧ closure L S = N :=
  Iff.refl _


theorem FG.cg {N : L.Substructure M} (h : N.FG) : N.CG := by
  /-
    L : FirstOrder.Language
    M : Type u_1
    inst✝ : L.Structure M
    N : L.Substructure M
    h : N.FG
    ⊢ N.CG
  -/
  obtain ⟨s, hf, rfl⟩ := fg_def.1 h
  /-
    case intro.intro
    L : FirstOrder.Language
    M : Type u_1
    inst✝ : L.Structure M
    s : Set M
    hf : s.Finite
    h : ((FirstOrder.Language.Substructure.closure L).toFun s).FG
    ⊢ ((FirstOrder.Language.Substructure.closure L).toFun s).CG
  -/
  exact ⟨s, hf.countable, rfl⟩
  /-
    🎉 no goals
  -/


theorem cg_iff_empty_or_exists_nat_generating_family {N : L.Substructure M} :
    N.CG ↔ N = (∅ : Set M) ∨ ∃ s : ℕ → M, closure L (range s) = N := by
  /-
    L : FirstOrder.Language
    M : Type u_1
    inst✝ : L.Structure M
    N : L.Substructure M
    ⊢ Iff N.CG (Or (Eq (↑N) EmptyCollection.emptyCollection) (Exists fun s => Eq ( …
  -/
  rw [cg_def]
  /-
    L : FirstOrder.Language
    M : Type u_1
    inst✝ : L.Structure M
    N : L.Substructure M
    ⊢ Iff (Exists fun S => And S.Countable (Eq ((FirstOrder.Language.Substructure. …
  -/
  constructor
    /-
      case mp
      L : FirstOrder.Language
      M : Type u_1
      inst✝ : L.Structure M
      N : L.Substructure M
      ⊢ (Exists fun S => And S.Countable (Eq ((FirstOrder.Language.Substructure.clos …
    -/
  · rintro ⟨S, Scount, hS⟩
    /-
      case mp.intro.intro
      L : FirstOrder.Language
      M : Type u_1
      inst✝ : L.Structure M
      N : L.Substructure M
      S : Set M
      Scount : S.Countable
      hS : Eq ((FirstOrder.Language.Substructure.closure L).toFun S) N
      ⊢ Or (Eq (↑N) EmptyCollection.emptyCollection) (Exists fun s => Eq ((FirstOrde …
    -/
    rcases eq_empty_or_nonempty (N : Set M) with h | h
      /-
        case mp.intro.intro.inl
        L : FirstOrder.Language
        M : Type u_1
        inst✝ : L.Structure M
        N : L.Substructure M
        S : Set M
        Scount : S.Countable
        hS : Eq ((FirstOrder.Language.Substructure.closure L).toFun S) N
        h : Eq (↑N) EmptyCollection.emptyCollection
        ⊢ Or (Eq (↑N) EmptyCollection.emptyCollection) (Exists fun s => Eq ((FirstOrde …
      -/
    · exact Or.intro_left _ h
      /-
        🎉 no goals
      -/
    obtain ⟨f, h'⟩ :=
      (Scount.union (Set.countable_singleton h.some)).exists_eq_range
        (singleton_nonempty h.some).inr
    /-
      case mp.intro.intro.inr.intro
      L : FirstOrder.Language
      M : Type u_1
      inst✝ : L.Structure M
      N : L.Substructure M
      S : Set M
      Scount : S.Countable
      hS : Eq ((FirstOrder.Language.Substructure.closure L).toFun S) N
      h : (↑N).Nonempty
      f : Nat → M
      h' : Eq (Union.union S (Singleton.singleton h.some)) (Set.range f)
      ⊢ Or (Eq (↑N) EmptyCollection.emptyCollection) (Exists fun s => Eq ((FirstOrde …
    -/
    refine Or.intro_right _ ⟨f, ?_⟩
    /-
      case mp.intro.intro.inr.intro
      L : FirstOrder.Language
      M : Type u_1
      inst✝ : L.Structure M
      N : L.Substructure M
      S : Set M
      Scount : S.Countable
      hS : Eq ((FirstOrder.Language.Substructure.closure L).toFun S) N
      h : (↑N).Nonempty
      f : Nat → M
      h' : Eq (Union.union S (Singleton.singleton h.some)) (Set.range f)
      ⊢ Eq ((FirstOrder.Language.Substructure.closure L).toFun (Set.range f)) N
    -/
    rw [← h', closure_union, hS, sup_eq_left, closure_le]
    /-
      case mp.intro.intro.inr.intro
      L : FirstOrder.Language
      M : Type u_1
      inst✝ : L.Structure M
      N : L.Substructure M
      S : Set M
      Scount : S.Countable
      hS : Eq ((FirstOrder.Language.Substructure.closure L).toFun S) N
      h : (↑N).Nonempty
      f : Nat → M
      h' : Eq (Union.union S (Singleton.singleton h.some)) (Set.range f)
      ⊢ HasSubset.Subset (Singleton.singleton h.some) ↑N
    -/
    exact singleton_subset_iff.2 h.some_mem
    /-
      🎉 no goals
    -/
    /-
      case mpr
      L : FirstOrder.Language
      M : Type u_1
      inst✝ : L.Structure M
      N : L.Substructure M
      ⊢ Or (Eq (↑N) EmptyCollection.emptyCollection) (Exists fun s => Eq ((FirstOrde …
    -/
  · intro h
    /-
      case mpr
      L : FirstOrder.Language
      M : Type u_1
      inst✝ : L.Structure M
      N : L.Substructure M
      h : Or (Eq (↑N) EmptyCollection.emptyCollection) (Exists fun s => Eq ((FirstOr …
      ⊢ Exists fun S => And S.Countable (Eq ((FirstOrder.Language.Substructure.closu …
    -/
    cases' h with h h
      /-
        case mpr.inl
        L : FirstOrder.Language
        M : Type u_1
        inst✝ : L.Structure M
        N : L.Substructure M
        h : Eq (↑N) EmptyCollection.emptyCollection
        ⊢ Exists fun S => And S.Countable (Eq ((FirstOrder.Language.Substructure.closu …
      -/
    · refine ⟨∅, countable_empty, closure_eq_of_le (empty_subset _) ?_⟩
      /-
        case mpr.inl
        L : FirstOrder.Language
        M : Type u_1
        inst✝ : L.Structure M
        N : L.Substructure M
        h : Eq (↑N) EmptyCollection.emptyCollection
        ⊢ LE.le N ((FirstOrder.Language.Substructure.closure L).toFun EmptyCollection. …
      -/
      rw [← SetLike.coe_subset_coe, h]
      /-
        case mpr.inl
        L : FirstOrder.Language
        M : Type u_1
        inst✝ : L.Structure M
        N : L.Substructure M
        h : Eq (↑N) EmptyCollection.emptyCollection
        ⊢ HasSubset.Subset EmptyCollection.emptyCollection ↑((FirstOrder.Language.Subs …
      -/
      exact empty_subset _
      /-
        🎉 no goals
      -/
      /-
        case mpr.inr
        L : FirstOrder.Language
        M : Type u_1
        inst✝ : L.Structure M
        N : L.Substructure M
        h : Exists fun s => Eq ((FirstOrder.Language.Substructure.closure L).toFun (Se …
        ⊢ Exists fun S => And S.Countable (Eq ((FirstOrder.Language.Substructure.closu …
      -/
    · obtain ⟨f, rfl⟩ := h
      /-
        case mpr.inr.intro
        L : FirstOrder.Language
        M : Type u_1
        inst✝ : L.Structure M
        f : Nat → M
        ⊢ Exists fun S => And S.Countable (Eq ((FirstOrder.Language.Substructure.closu …
      -/
      exact ⟨range f, countable_range _, rfl⟩
      /-
        🎉 no goals
      -/


theorem cg_bot : (⊥ : L.Substructure M).CG :=
  fg_bot.cg


theorem cg_closure {s : Set M} (hs : s.Countable) : CG (closure L s) :=
  ⟨s, hs, rfl⟩


theorem cg_closure_singleton (x : M) : CG (closure L ({x} : Set M)) :=
  (fg_closure_singleton x).cg


theorem CG.sup {N₁ N₂ : L.Substructure M} (hN₁ : N₁.CG) (hN₂ : N₂.CG) : (N₁ ⊔ N₂).CG :=
  let ⟨t₁, ht₁⟩ := cg_def.1 hN₁
  let ⟨t₂, ht₂⟩ := cg_def.1 hN₂
                                           /-
                                             L : FirstOrder.Language
                                             M : Type u_1
                                             inst✝ : L.Structure M
                                             N₁ N₂ : L.Substructure M
                                             hN₁ : N₁.CG
                                             hN₂ : N₂.CG
                                             t₁ : Set M
                                             ht₁ : And t₁.Countable (Eq ((FirstOrder.Language.Substructure.closure L).toFun …
                                             t₂ : Set M
                                             ht₂ : And t₂.Countable (Eq ((FirstOrder.Language.Substructure.closure L).toFun …
                                             ⊢ Eq ((FirstOrder.Language.Substructure.closure L).toFun (Union.union t₁ t₂))  …
                                           -/
  cg_def.2 ⟨t₁ ∪ t₂, ht₁.1.union ht₂.1, by rw [closure_union, ht₁.2, ht₂.2]⟩
                                           /-
                                             🎉 no goals
                                           -/


theorem CG.map {N : Type*} [L.Structure N] (f : M →[L] N) {s : L.Substructure M} (hs : s.CG) :
    (s.map f).CG :=
  let ⟨t, ht⟩ := cg_def.1 hs
                                     /-
                                       L : FirstOrder.Language
                                       M : Type u_1
                                       inst✝¹ : L.Structure M
                                       N : Type u_2
                                       inst✝ : L.Structure N
                                       f : L.Hom M N
                                       s : L.Substructure M
                                       hs : s.CG
                                       t : Set M
                                       ht : And t.Countable (Eq ((FirstOrder.Language.Substructure.closure L).toFun t …
                                       ⊢ Eq ((FirstOrder.Language.Substructure.closure L).toFun (Set.image (⇑f) t)) ( …
                                     -/
  cg_def.2 ⟨f '' t, ht.1.image _, by rw [closure_image, ht.2]⟩
                                     /-
                                       🎉 no goals
                                     -/


theorem CG.of_map_embedding {N : Type*} [L.Structure N] (f : M ↪[L] N) {s : L.Substructure M}
    (hs : (s.map f.toHom).CG) : s.CG := by
  /-
    L : FirstOrder.Language
    M : Type u_1
    inst✝¹ : L.Structure M
    N : Type u_2
    inst✝ : L.Structure N
    f : L.Embedding M N
    s : L.Substructure M
    hs : (FirstOrder.Language.Substructure.map f.toHom s).CG
    ⊢ s.CG
  -/
  rcases hs with ⟨t, h1, h2⟩
  /-
    case intro.intro
    L : FirstOrder.Language
    M : Type u_1
    inst✝¹ : L.Structure M
    N : Type u_2
    inst✝ : L.Structure N
    f : L.Embedding M N
    s : L.Substructure M
    t : Set N
    h1 : t.Countable
    h2 : Eq ((FirstOrder.Language.Substructure.closure L).toFun t) (FirstOrder.Lan …
    ⊢ s.CG
  -/
  rw [cg_def]
  /-
    case intro.intro
    L : FirstOrder.Language
    M : Type u_1
    inst✝¹ : L.Structure M
    N : Type u_2
    inst✝ : L.Structure N
    f : L.Embedding M N
    s : L.Substructure M
    t : Set N
    h1 : t.Countable
    h2 : Eq ((FirstOrder.Language.Substructure.closure L).toFun t) (FirstOrder.Lan …
    ⊢ Exists fun S => And S.Countable (Eq ((FirstOrder.Language.Substructure.closu …
  -/
  refine ⟨f ⁻¹' t, h1.preimage f.injective, ?_⟩
  /-
    case intro.intro
    L : FirstOrder.Language
    M : Type u_1
    inst✝¹ : L.Structure M
    N : Type u_2
    inst✝ : L.Structure N
    f : L.Embedding M N
    s : L.Substructure M
    t : Set N
    h1 : t.Countable
    h2 : Eq ((FirstOrder.Language.Substructure.closure L).toFun t) (FirstOrder.Lan …
    ⊢ Eq ((FirstOrder.Language.Substructure.closure L).toFun (Set.preimage (⇑f) t) …
  -/
  have hf : Function.Injective f.toHom := f.injective
  /-
    case intro.intro
    L : FirstOrder.Language
    M : Type u_1
    inst✝¹ : L.Structure M
    N : Type u_2
    inst✝ : L.Structure N
    f : L.Embedding M N
    s : L.Substructure M
    t : Set N
    h1 : t.Countable
    h2 : Eq ((FirstOrder.Language.Substructure.closure L).toFun t) (FirstOrder.Lan …
    hf : Function.Injective ⇑f.toHom
    ⊢ Eq ((FirstOrder.Language.Substructure.closure L).toFun (Set.preimage (⇑f) t) …
  -/
  refine map_injective_of_injective hf ?_
  /-
    case intro.intro
    L : FirstOrder.Language
    M : Type u_1
    inst✝¹ : L.Structure M
    N : Type u_2
    inst✝ : L.Structure N
    f : L.Embedding M N
    s : L.Substructure M
    t : Set N
    h1 : t.Countable
    h2 : Eq ((FirstOrder.Language.Substructure.closure L).toFun t) (FirstOrder.Lan …
    hf : Function.Injective ⇑f.toHom
    ⊢ Eq (FirstOrder.Language.Substructure.map f.toHom ((FirstOrder.Language.Subst …
  -/
  rw [← h2, map_closure, Embedding.coe_toHom, image_preimage_eq_of_subset]
  /-
    case intro.intro
    L : FirstOrder.Language
    M : Type u_1
    inst✝¹ : L.Structure M
    N : Type u_2
    inst✝ : L.Structure N
    f : L.Embedding M N
    s : L.Substructure M
    t : Set N
    h1 : t.Countable
    h2 : Eq ((FirstOrder.Language.Substructure.closure L).toFun t) (FirstOrder.Lan …
    hf : Function.Injective ⇑f.toHom
    ⊢ HasSubset.Subset t (Set.range ⇑f)
  -/
  intro x hx
  /-
    case intro.intro
    L : FirstOrder.Language
    M : Type u_1
    inst✝¹ : L.Structure M
    N : Type u_2
    inst✝ : L.Structure N
    f : L.Embedding M N
    s : L.Substructure M
    t : Set N
    h1 : t.Countable
    h2 : Eq ((FirstOrder.Language.Substructure.closure L).toFun t) (FirstOrder.Lan …
    hf : Function.Injective ⇑f.toHom
    x : N
    hx : Membership.mem t x
    ⊢ Membership.mem (Set.range ⇑f) x
  -/
  have h' := subset_closure (L := L) hx
  /-
    case intro.intro
    L : FirstOrder.Language
    M : Type u_1
    inst✝¹ : L.Structure M
    N : Type u_2
    inst✝ : L.Structure N
    f : L.Embedding M N
    s : L.Substructure M
    t : Set N
    h1 : t.Countable
    h2 : Eq ((FirstOrder.Language.Substructure.closure L).toFun t) (FirstOrder.Lan …
    hf : Function.Injective ⇑f.toHom
    x : N
    hx : Membership.mem t x
    h' : Membership.mem (↑((FirstOrder.Language.Substructure.closure L).toFun t)) x
    ⊢ Membership.mem (Set.range ⇑f) x
  -/
  rw [h2] at h'
  /-
    case intro.intro
    L : FirstOrder.Language
    M : Type u_1
    inst✝¹ : L.Structure M
    N : Type u_2
    inst✝ : L.Structure N
    f : L.Embedding M N
    s : L.Substructure M
    t : Set N
    h1 : t.Countable
    h2 : Eq ((FirstOrder.Language.Substructure.closure L).toFun t) (FirstOrder.Lan …
    hf : Function.Injective ⇑f.toHom
    x : N
    hx : Membership.mem t x
    h' : Membership.mem (↑(FirstOrder.Language.Substructure.map f.toHom s)) x
    ⊢ Membership.mem (Set.range ⇑f) x
  -/
  exact Hom.map_le_range h'
  /-
    🎉 no goals
  -/


theorem cg_iff_countable [Countable (Σl, L.Functions l)] {s : L.Substructure M} :
    s.CG ↔ Countable s := by
  /-
    L : FirstOrder.Language
    M : Type u_1
    inst✝¹ : L.Structure M
    inst✝ : Countable (Sigma fun l => L.Functions l)
    s : L.Substructure M
    ⊢ Iff s.CG (Countable (Subtype fun x => Membership.mem s x))
  -/
  refine ⟨?_, fun h => ⟨s, h.to_set, s.closure_eq⟩⟩
  /-
    L : FirstOrder.Language
    M : Type u_1
    inst✝¹ : L.Structure M
    inst✝ : Countable (Sigma fun l => L.Functions l)
    s : L.Substructure M
    ⊢ s.CG → Countable (Subtype fun x => Membership.mem s x)
  -/
  rintro ⟨s, h, rfl⟩
  /-
    case intro.intro
    L : FirstOrder.Language
    M : Type u_1
    inst✝¹ : L.Structure M
    inst✝ : Countable (Sigma fun l => L.Functions l)
    s : Set M
    h : s.Countable
    ⊢ Countable (Subtype fun x => Membership.mem ((FirstOrder.Language.Substructur …
  -/
  exact h.substructure_closure L
  /-
    🎉 no goals
  -/


theorem cg_of_countable {s : L.Substructure M} [h : Countable s] : s.CG :=
  ⟨s, h.to_set, s.closure_eq⟩


/-- A structure is finitely generated if it is the closure of a finite subset. -/
class FG : Prop where
  out : (⊤ : L.Substructure M).FG


/-- A structure is countably generated if it is the closure of a countable subset. -/
class CG : Prop where
  out : (⊤ : L.Substructure M).CG


theorem fg_def : FG L M ↔ (⊤ : L.Substructure M).FG :=
  ⟨fun h => h.1, fun h => ⟨h⟩⟩


/-- An equivalent expression of `Structure.FG` in terms of `Set.Finite` instead of `Finset`. -/
theorem fg_iff : FG L M ↔ ∃ S : Set M, S.Finite ∧ closure L S = (⊤ : L.Substructure M) := by
  /-
    L : FirstOrder.Language
    M : Type u_1
    inst✝ : L.Structure M
    ⊢ Iff (FirstOrder.Language.Structure.FG L M) (Exists fun S => And S.Finite (Eq …
  -/
  rw [fg_def, Substructure.fg_def]
  /-
    🎉 no goals
  -/


theorem FG.range {N : Type*} [L.Structure N] (h : FG L M) (f : M →[L] N) : f.range.FG := by
  /-
    L : FirstOrder.Language
    M : Type u_1
    inst✝¹ : L.Structure M
    N : Type u_2
    inst✝ : L.Structure N
    h : FirstOrder.Language.Structure.FG L M
    f : L.Hom M N
    ⊢ f.range.FG
  -/
  rw [Hom.range_eq_map]
  /-
    L : FirstOrder.Language
    M : Type u_1
    inst✝¹ : L.Structure M
    N : Type u_2
    inst✝ : L.Structure N
    h : FirstOrder.Language.Structure.FG L M
    f : L.Hom M N
    ⊢ (FirstOrder.Language.Substructure.map f Top.top).FG
  -/
  exact (fg_def.1 h).map f
  /-
    🎉 no goals
  -/


theorem FG.map_of_surjective {N : Type*} [L.Structure N] (h : FG L M) (f : M →[L] N)
    (hs : Function.Surjective f) : FG L N := by
  /-
    L : FirstOrder.Language
    M : Type u_1
    inst✝¹ : L.Structure M
    N : Type u_2
    inst✝ : L.Structure N
    h : FirstOrder.Language.Structure.FG L M
    f : L.Hom M N
    hs : Function.Surjective ⇑f
    ⊢ FirstOrder.Language.Structure.FG L N
  -/
  rw [← Hom.range_eq_top] at hs
  /-
    L : FirstOrder.Language
    M : Type u_1
    inst✝¹ : L.Structure M
    N : Type u_2
    inst✝ : L.Structure N
    h : FirstOrder.Language.Structure.FG L M
    f : L.Hom M N
    hs : Eq f.range Top.top
    ⊢ FirstOrder.Language.Structure.FG L N
  -/
  rw [fg_def, ← hs]
  /-
    L : FirstOrder.Language
    M : Type u_1
    inst✝¹ : L.Structure M
    N : Type u_2
    inst✝ : L.Structure N
    h : FirstOrder.Language.Structure.FG L M
    f : L.Hom M N
    hs : Eq f.range Top.top
    ⊢ f.range.FG
  -/
  exact h.range f
  /-
    🎉 no goals
  -/


theorem FG.countable_hom (N : Type*) [L.Structure N] [Countable N] (h : FG L M) :
    Countable (M →[L] N) := by
  /-
    L : FirstOrder.Language
    M : Type u_1
    inst✝² : L.Structure M
    N : Type u_2
    inst✝¹ : L.Structure N
    inst✝ : Countable N
    h : FirstOrder.Language.Structure.FG L M
    ⊢ Countable (L.Hom M N)
  -/
  let ⟨S, finite_S, closure_S⟩ := fg_iff.1 h
  let g : (M →[L] N) → (S → N) :=
    fun f ↦ f ∘ (↑)
  have g_inj : Function.Injective g := by
    intro f f' h
    apply Hom.eq_of_eqOn_dense closure_S
    intro x x_in_S
    exact congr_fun h ⟨x, x_in_S⟩
  /-
    L : FirstOrder.Language
    M : Type u_1
    inst✝² : L.Structure M
    N : Type u_2
    inst✝¹ : L.Structure N
    inst✝ : Countable N
    h : FirstOrder.Language.Structure.FG L M
    S : Set M
    finite_S : S.Finite
    closure_S : Eq ((FirstOrder.Language.Substructure.closure L).toFun S) Top.top
    g : L.Hom M N → ↑S → N := fun f => Function.comp (⇑f) Subtype.val
    g_inj : Function.Injective g
    ⊢ Countable (L.Hom M N)
  -/
  have : Finite ↑S := (S.finite_coe_iff).2 finite_S
  /-
    L : FirstOrder.Language
    M : Type u_1
    inst✝² : L.Structure M
    N : Type u_2
    inst✝¹ : L.Structure N
    inst✝ : Countable N
    h : FirstOrder.Language.Structure.FG L M
    S : Set M
    finite_S : S.Finite
    closure_S : Eq ((FirstOrder.Language.Substructure.closure L).toFun S) Top.top
    g : L.Hom M N → ↑S → N := fun f => Function.comp (⇑f) Subtype.val
    g_inj : Function.Injective g
    this : Finite ↑S
    ⊢ Countable (L.Hom M N)
  -/
  exact Function.Embedding.countable ⟨g, g_inj⟩
  /-
    🎉 no goals
  -/


instance FG.instCountable_hom (N : Type*) [L.Structure N] [Countable N] [h : FG L M] :
    Countable (M →[L] N) :=
  FG.countable_hom N h


theorem FG.countable_embedding (N : Type*) [L.Structure N] [Countable N] (_ : FG L M) :
    Countable (M ↪[L] N) :=
  Function.Embedding.countable ⟨Embedding.toHom, Embedding.toHom_injective⟩


instance Fg.instCountable_embedding (N : Type*) [L.Structure N]
    [Countable N] [h : FG L M] : Countable (M ↪[L] N) :=
  FG.countable_embedding N h


theorem FG.of_finite [Finite M] : FG L M := by
  /-
    L : FirstOrder.Language
    M : Type u_1
    inst✝¹ : L.Structure M
    inst✝ : Finite M
    ⊢ FirstOrder.Language.Structure.FG L M
  -/
  simp only [fg_def, Substructure.FG.of_finite, topEquiv.toEquiv.finite_iff]
  /-
    🎉 no goals
  -/


theorem FG.finite [L.IsRelational] (h : FG L M) : Finite M :=
  Finite.of_finite_univ (Substructure.FG.finite (fg_def.1 h))


theorem fg_iff_finite [L.IsRelational] : FG L M ↔ Finite M :=
  ⟨FG.finite, fun _ => FG.of_finite⟩


theorem cg_def : CG L M ↔ (⊤ : L.Substructure M).CG :=
  ⟨fun h => h.1, fun h => ⟨h⟩⟩


/-- An equivalent expression of `Structure.cg`. -/
theorem cg_iff : CG L M ↔ ∃ S : Set M, S.Countable ∧ closure L S = (⊤ : L.Substructure M) := by
  /-
    L : FirstOrder.Language
    M : Type u_1
    inst✝ : L.Structure M
    ⊢ Iff (FirstOrder.Language.Structure.CG L M) (Exists fun S => And S.Countable  …
  -/
  rw [cg_def, Substructure.cg_def]
  /-
    🎉 no goals
  -/


theorem CG.range {N : Type*} [L.Structure N] (h : CG L M) (f : M →[L] N) : f.range.CG := by
  /-
    L : FirstOrder.Language
    M : Type u_1
    inst✝¹ : L.Structure M
    N : Type u_2
    inst✝ : L.Structure N
    h : FirstOrder.Language.Structure.CG L M
    f : L.Hom M N
    ⊢ f.range.CG
  -/
  rw [Hom.range_eq_map]
  /-
    L : FirstOrder.Language
    M : Type u_1
    inst✝¹ : L.Structure M
    N : Type u_2
    inst✝ : L.Structure N
    h : FirstOrder.Language.Structure.CG L M
    f : L.Hom M N
    ⊢ (FirstOrder.Language.Substructure.map f Top.top).CG
  -/
  exact (cg_def.1 h).map f
  /-
    🎉 no goals
  -/


theorem CG.map_of_surjective {N : Type*} [L.Structure N] (h : CG L M) (f : M →[L] N)
    (hs : Function.Surjective f) : CG L N := by
  /-
    L : FirstOrder.Language
    M : Type u_1
    inst✝¹ : L.Structure M
    N : Type u_2
    inst✝ : L.Structure N
    h : FirstOrder.Language.Structure.CG L M
    f : L.Hom M N
    hs : Function.Surjective ⇑f
    ⊢ FirstOrder.Language.Structure.CG L N
  -/
  rw [← Hom.range_eq_top] at hs
  /-
    L : FirstOrder.Language
    M : Type u_1
    inst✝¹ : L.Structure M
    N : Type u_2
    inst✝ : L.Structure N
    h : FirstOrder.Language.Structure.CG L M
    f : L.Hom M N
    hs : Eq f.range Top.top
    ⊢ FirstOrder.Language.Structure.CG L N
  -/
  rw [cg_def, ← hs]
  /-
    L : FirstOrder.Language
    M : Type u_1
    inst✝¹ : L.Structure M
    N : Type u_2
    inst✝ : L.Structure N
    h : FirstOrder.Language.Structure.CG L M
    f : L.Hom M N
    hs : Eq f.range Top.top
    ⊢ f.range.CG
  -/
  exact h.range f
  /-
    🎉 no goals
  -/


theorem cg_iff_countable [Countable (Σl, L.Functions l)] : CG L M ↔ Countable M := by
  /-
    L : FirstOrder.Language
    M : Type u_1
    inst✝¹ : L.Structure M
    inst✝ : Countable (Sigma fun l => L.Functions l)
    ⊢ Iff (FirstOrder.Language.Structure.CG L M) (Countable M)
  -/
  rw [cg_def, Substructure.cg_iff_countable, topEquiv.toEquiv.countable_iff]
  /-
    🎉 no goals
  -/


theorem cg_of_countable [Countable M] : CG L M := by
  /-
    L : FirstOrder.Language
    M : Type u_1
    inst✝¹ : L.Structure M
    inst✝ : Countable M
    ⊢ FirstOrder.Language.Structure.CG L M
  -/
  simp only [cg_def, Substructure.cg_of_countable, topEquiv.toEquiv.countable_iff]
  /-
    🎉 no goals
  -/


theorem FG.cg (h : FG L M) : CG L M :=
  cg_def.2 (fg_def.1 h).cg


instance (priority := 100) cg_of_fg [h : FG L M] : CG L M :=
  h.cg


theorem Equiv.fg_iff {N : Type*} [L.Structure N] (f : M ≃[L] N) :
    Structure.FG L M ↔ Structure.FG L N :=
  ⟨fun h => h.map_of_surjective f.toHom f.toEquiv.surjective, fun h =>
    h.map_of_surjective f.symm.toHom f.toEquiv.symm.surjective⟩


theorem Substructure.fg_iff_structure_fg (S : L.Substructure M) : S.FG ↔ Structure.FG L S := by
  /-
    L : FirstOrder.Language
    M : Type u_1
    inst✝ : L.Structure M
    S : L.Substructure M
    ⊢ Iff S.FG (FirstOrder.Language.Structure.FG L (Subtype fun x => Membership.me …
  -/
  rw [Structure.fg_def]
  /-
    L : FirstOrder.Language
    M : Type u_1
    inst✝ : L.Structure M
    S : L.Substructure M
    ⊢ Iff S.FG Top.top.FG
  -/
  refine ⟨fun h => FG.of_map_embedding S.subtype ?_, fun h => ?_⟩
    /-
      case refine_1
      L : FirstOrder.Language
      M : Type u_1
      inst✝ : L.Structure M
      S : L.Substructure M
      h : S.FG
      ⊢ (FirstOrder.Language.Substructure.map S.subtype.toHom Top.top).FG
    -/
  · rw [← Hom.range_eq_map, range_subtype]
    /-
      case refine_1
      L : FirstOrder.Language
      M : Type u_1
      inst✝ : L.Structure M
      S : L.Substructure M
      h : S.FG
      ⊢ S.FG
    -/
    exact h
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      L : FirstOrder.Language
      M : Type u_1
      inst✝ : L.Structure M
      S : L.Substructure M
      h : Top.top.FG
      ⊢ S.FG
    -/
  · have h := h.map S.subtype.toHom
    /-
      case refine_2
      L : FirstOrder.Language
      M : Type u_1
      inst✝ : L.Structure M
      S : L.Substructure M
      h✝ : Top.top.FG
      h : (FirstOrder.Language.Substructure.map S.subtype.toHom Top.top).FG
      ⊢ S.FG
    -/
    rw [← Hom.range_eq_map, range_subtype] at h
    /-
      case refine_2
      L : FirstOrder.Language
      M : Type u_1
      inst✝ : L.Structure M
      S : L.Substructure M
      h✝ : Top.top.FG
      h : S.FG
      ⊢ S.FG
    -/
    exact h
    /-
      🎉 no goals
    -/


theorem Equiv.cg_iff {N : Type*} [L.Structure N] (f : M ≃[L] N) :
    Structure.CG L M ↔ Structure.CG L N :=
  ⟨fun h => h.map_of_surjective f.toHom f.toEquiv.surjective, fun h =>
    h.map_of_surjective f.symm.toHom f.toEquiv.symm.surjective⟩


theorem Substructure.cg_iff_structure_cg (S : L.Substructure M) : S.CG ↔ Structure.CG L S := by
  /-
    L : FirstOrder.Language
    M : Type u_1
    inst✝ : L.Structure M
    S : L.Substructure M
    ⊢ Iff S.CG (FirstOrder.Language.Structure.CG L (Subtype fun x => Membership.me …
  -/
  rw [Structure.cg_def]
  /-
    L : FirstOrder.Language
    M : Type u_1
    inst✝ : L.Structure M
    S : L.Substructure M
    ⊢ Iff S.CG Top.top.CG
  -/
  refine ⟨fun h => CG.of_map_embedding S.subtype ?_, fun h => ?_⟩
    /-
      case refine_1
      L : FirstOrder.Language
      M : Type u_1
      inst✝ : L.Structure M
      S : L.Substructure M
      h : S.CG
      ⊢ (FirstOrder.Language.Substructure.map S.subtype.toHom Top.top).CG
    -/
  · rw [← Hom.range_eq_map, range_subtype]
    /-
      case refine_1
      L : FirstOrder.Language
      M : Type u_1
      inst✝ : L.Structure M
      S : L.Substructure M
      h : S.CG
      ⊢ S.CG
    -/
    exact h
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      L : FirstOrder.Language
      M : Type u_1
      inst✝ : L.Structure M
      S : L.Substructure M
      h : Top.top.CG
      ⊢ S.CG
    -/
  · have h := h.map S.subtype.toHom
    /-
      case refine_2
      L : FirstOrder.Language
      M : Type u_1
      inst✝ : L.Structure M
      S : L.Substructure M
      h✝ : Top.top.CG
      h : (FirstOrder.Language.Substructure.map S.subtype.toHom Top.top).CG
      ⊢ S.CG
    -/
    rw [← Hom.range_eq_map, range_subtype] at h
    /-
      case refine_2
      L : FirstOrder.Language
      M : Type u_1
      inst✝ : L.Structure M
      S : L.Substructure M
      h✝ : Top.top.CG
      h : S.CG
      ⊢ S.CG
    -/
    exact h
    /-
      🎉 no goals
    -/


theorem Substructure.countable_fg_substructures_of_countable [Countable M] :
    Countable { S : L.Substructure M // S.FG } := by
  let g : { S : L.Substructure M // S.FG } → Finset M :=
    fun S ↦ Exists.choose S.prop
  have g_inj : Function.Injective g := by
    intro S S' h
    apply Subtype.eq
    rw [(Exists.choose_spec S.prop).symm, (Exists.choose_spec S'.prop).symm]
    exact congr_arg ((closure L) ∘ Finset.toSet) h
  /-
    L : FirstOrder.Language
    M : Type u_1
    inst✝¹ : L.Structure M
    inst✝ : Countable M
    g : (Subtype fun S => S.FG) → Finset M := fun S => Exists.choose ⋯
    g_inj : Function.Injective g
    ⊢ Countable (Subtype fun S => S.FG)
  -/
  exact Function.Embedding.countable ⟨g, g_inj⟩
  /-
    🎉 no goals
  -/


instance Substructure.instCountable_fg_substructures_of_countable [Countable M] :
    Countable { S : L.Substructure M // S.FG } :=
  countable_fg_substructures_of_countable


