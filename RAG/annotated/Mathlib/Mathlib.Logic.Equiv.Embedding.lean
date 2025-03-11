/-- Embeddings from a sum type are equivalent to two separate embeddings with disjoint ranges. -/
def sumEmbeddingEquivProdEmbeddingDisjoint {α β γ : Type*} :
    (α ⊕ β ↪ γ) ≃ { f : (α ↪ γ) × (β ↪ γ) // Disjoint (Set.range f.1) (Set.range f.2) } where
  toFun f :=
    ⟨(inl.trans f, inr.trans f), by
      /-
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        f : Function.Embedding (Sum α β) γ
        ⊢ Disjoint (Set.range ⇑{ fst := Function.Embedding.inl.trans f, snd := Functio …
      -/
      rw [Set.disjoint_left]
      /-
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        f : Function.Embedding (Sum α β) γ
        ⊢ ∀ ⦃a : γ⦄, Membership.mem (Set.range ⇑{ fst := Function.Embedding.inl.trans  …
      -/
      rintro _ ⟨a, h⟩ ⟨b, rfl⟩
      /-
        case intro.intro
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        f : Function.Embedding (Sum α β) γ
        a : α
        b : β
        h : Eq ({ fst := Function.Embedding.inl.trans f, snd := Function.Embedding.inr …
        ⊢ False
      -/
      simp only [trans_apply, inl_apply, inr_apply] at h
      /-
        case intro.intro
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        f : Function.Embedding (Sum α β) γ
        a : α
        b : β
        h : Eq ((Function.Embedding.inl.trans f) a) ((Function.Embedding.inr.trans f) b)
        ⊢ False
      -/
      have : Sum.inl a = Sum.inr b := f.injective h
      /-
        case intro.intro
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        f : Function.Embedding (Sum α β) γ
        a : α
        b : β
        h : Eq ((Function.Embedding.inl.trans f) a) ((Function.Embedding.inr.trans f) b)
        this : Eq (Sum.inl a) (Sum.inr b)
        ⊢ False
      -/
      simp only [reduceCtorEq] at this⟩
      /-
        🎉 no goals
      -/
  invFun := fun ⟨⟨f, g⟩, disj⟩ =>
    ⟨fun x =>
      match x with
      | Sum.inl a => f a
      | Sum.inr b => g b, by
      /-
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        x✝ : Subtype fun f => Disjoint (Set.range ⇑f.1) (Set.range ⇑f.2)
        f : Function.Embedding α γ
        g : Function.Embedding β γ
        disj : Disjoint (Set.range ⇑{ fst := f, snd := g }.1) (Set.range ⇑{ fst := f,  …
        ⊢ Function.Injective fun x => Equiv.sumEmbeddingEquivProdEmbeddingDisjoint.mat …
      -/
      rintro (a₁ | b₁) (a₂ | b₂) f_eq <;>
        /-
          case inl.inl
          α : Type u_1
          β : Type u_2
          γ : Type u_3
          x✝ : Subtype fun f => Disjoint (Set.range ⇑f.1) (Set.range ⇑f.2)
          f : Function.Embedding α γ
          g : Function.Embedding β γ
          disj : Disjoint (Set.range ⇑{ fst := f, snd := g }.1) (Set.range ⇑{ fst := f,  …
          a₁ a₂ : α
          f_eq : Eq ((fun x => Equiv.sumEmbeddingEquivProdEmbeddingDisjoint.match_1 (fun …
          ⊢ Eq (Sum.inl a₁) (Sum.inl a₂)
        -/
        simp only [Equiv.coe_fn_symm_mk, Sum.elim_inl, Sum.elim_inr] at f_eq
        /-
          case inl.inl
          α : Type u_1
          β : Type u_2
          γ : Type u_3
          x✝ : Subtype fun f => Disjoint (Set.range ⇑f.1) (Set.range ⇑f.2)
          f : Function.Embedding α γ
          g : Function.Embedding β γ
          disj : Disjoint (Set.range ⇑{ fst := f, snd := g }.1) (Set.range ⇑{ fst := f,  …
          a₁ a₂ : α
          f_eq : Eq (f a₁) (f a₂)
          ⊢ Eq (Sum.inl a₁) (Sum.inl a₂)
        -/
      · rw [f.injective f_eq]
        /-
          🎉 no goals
        -/
        /-
          case inl.inr
          α : Type u_1
          β : Type u_2
          γ : Type u_3
          x✝ : Subtype fun f => Disjoint (Set.range ⇑f.1) (Set.range ⇑f.2)
          f : Function.Embedding α γ
          g : Function.Embedding β γ
          disj : Disjoint (Set.range ⇑{ fst := f, snd := g }.1) (Set.range ⇑{ fst := f,  …
          a₁ : α
          b₂ : β
          f_eq : Eq (f a₁) (g b₂)
          ⊢ Eq (Sum.inl a₁) (Sum.inr b₂)
        -/
      · exfalso
        /-
          case inl.inr
          α : Type u_1
          β : Type u_2
          γ : Type u_3
          x✝ : Subtype fun f => Disjoint (Set.range ⇑f.1) (Set.range ⇑f.2)
          f : Function.Embedding α γ
          g : Function.Embedding β γ
          disj : Disjoint (Set.range ⇑{ fst := f, snd := g }.1) (Set.range ⇑{ fst := f,  …
          a₁ : α
          b₂ : β
          f_eq : Eq (f a₁) (g b₂)
          ⊢ False
        -/
        exact disj.le_bot ⟨⟨a₁, f_eq⟩, ⟨b₂, by simp [f_eq]⟩⟩
        /-
          🎉 no goals
        -/
        /-
          case inr.inl
          α : Type u_1
          β : Type u_2
          γ : Type u_3
          x✝ : Subtype fun f => Disjoint (Set.range ⇑f.1) (Set.range ⇑f.2)
          f : Function.Embedding α γ
          g : Function.Embedding β γ
          disj : Disjoint (Set.range ⇑{ fst := f, snd := g }.1) (Set.range ⇑{ fst := f,  …
          b₁ : β
          a₂ : α
          f_eq : Eq (g b₁) (f a₂)
          ⊢ Eq (Sum.inr b₁) (Sum.inl a₂)
        -/
      · exfalso
        /-
          case inr.inl
          α : Type u_1
          β : Type u_2
          γ : Type u_3
          x✝ : Subtype fun f => Disjoint (Set.range ⇑f.1) (Set.range ⇑f.2)
          f : Function.Embedding α γ
          g : Function.Embedding β γ
          disj : Disjoint (Set.range ⇑{ fst := f, snd := g }.1) (Set.range ⇑{ fst := f,  …
          b₁ : β
          a₂ : α
          f_eq : Eq (g b₁) (f a₂)
          ⊢ False
        -/
        exact disj.le_bot ⟨⟨a₂, rfl⟩, ⟨b₁, f_eq⟩⟩
        /-
          🎉 no goals
        -/
        /-
          case inr.inr
          α : Type u_1
          β : Type u_2
          γ : Type u_3
          x✝ : Subtype fun f => Disjoint (Set.range ⇑f.1) (Set.range ⇑f.2)
          f : Function.Embedding α γ
          g : Function.Embedding β γ
          disj : Disjoint (Set.range ⇑{ fst := f, snd := g }.1) (Set.range ⇑{ fst := f,  …
          b₁ b₂ : β
          f_eq : Eq (g b₁) (g b₂)
          ⊢ Eq (Sum.inr b₁) (Sum.inr b₂)
        -/
      · rw [g.injective f_eq]⟩
        /-
          🎉 no goals
        -/
  left_inv f := by
    /-
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      f : Function.Embedding (Sum α β) γ
      ⊢ Eq ((fun x => Equiv.sumEmbeddingEquivProdEmbeddingDisjoint.match_2 (fun x => …
    -/
    dsimp only
    /-
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      f : Function.Embedding (Sum α β) γ
      ⊢ Eq { toFun := fun x => Equiv.sumEmbeddingEquivProdEmbeddingDisjoint.match_1  …
    -/
    ext x
    /-
      case h
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      f : Function.Embedding (Sum α β) γ
      x : Sum α β
      ⊢ Eq ({ toFun := fun x => Equiv.sumEmbeddingEquivProdEmbeddingDisjoint.match_1 …
    -/
                /-
                  🎉 no goals
                -/
    cases x <;> simp!
                /-
                  🎉 no goals
                -/
  right_inv := fun ⟨⟨f, g⟩, _⟩ => by
    /-
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      x✝ : Subtype fun f => Disjoint (Set.range ⇑f.1) (Set.range ⇑f.2)
      f : Function.Embedding α γ
      g : Function.Embedding β γ
      property✝ : Disjoint (Set.range ⇑{ fst := f, snd := g }.1) (Set.range ⇑{ fst : …
      ⊢ Eq ((fun f => ⟨{ fst := Function.Embedding.inl.trans f, snd := Function.Embe …
    -/
    simp only [Prod.mk.inj_iff]
    /-
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      x✝ : Subtype fun f => Disjoint (Set.range ⇑f.1) (Set.range ⇑f.2)
      f : Function.Embedding α γ
      g : Function.Embedding β γ
      property✝ : Disjoint (Set.range ⇑{ fst := f, snd := g }.1) (Set.range ⇑{ fst : …
      ⊢ Eq ⟨{ fst := Function.Embedding.inl.trans { toFun := fun x => Equiv.sumEmbed …
    -/
    constructor
    /-
      🎉 no goals
    -/


/-- Embeddings whose range lies within a set are equivalent to embeddings to that set.
This is `Function.Embedding.codRestrict` as an equiv. -/
def codRestrict (α : Type*) {β : Type*} (bs : Set β) :
    { f : α ↪ β // ∀ a, f a ∈ bs } ≃
      (α ↪ bs) where
  toFun f := (f : α ↪ β).codRestrict bs f.prop
  invFun f := ⟨f.trans (Function.Embedding.subtype _), fun a => (f a).prop⟩
                   /-
                     α : Type u_1
                     β : Type u_2
                     bs : Set β
                     x : Subtype fun f => ∀ (a : α), Membership.mem bs (f a)
                     ⊢ Eq ((fun f => ⟨f.trans (Function.Embedding.subtype fun x => Membership.mem b …
                   -/
  left_inv x := by ext; rfl
                        /-
                          🎉 no goals
                        -/
                    /-
                      α : Type u_1
                      β : Type u_2
                      bs : Set β
                      x : Function.Embedding α ↑bs
                      ⊢ Eq ((fun f => Function.Embedding.codRestrict bs ↑f ⋯) ((fun f => ⟨f.trans (F …
                    -/
  right_inv x := by ext; rfl
                         /-
                           🎉 no goals
                         -/


/-- Pairs of embeddings with disjoint ranges are equivalent to a dependent sum of embeddings,
in which the second embedding cannot take values in the range of the first. -/
def prodEmbeddingDisjointEquivSigmaEmbeddingRestricted {α β γ : Type*} :
    { f : (α ↪ γ) × (β ↪ γ) // Disjoint (Set.range f.1) (Set.range f.2) } ≃
      Σf : α ↪ γ, β ↪ ↥(Set.range f)ᶜ :=
  (subtypeProdEquivSigmaSubtype fun (a : α ↪ γ) (b : β ↪ _) =>
        Disjoint (Set.range a) (Set.range b)).trans <|
    Equiv.sigmaCongrRight fun a =>
      (subtypeEquivProp <| by
            /-
              α : Type u_1
              β : Type u_2
              γ : Type u_3
              a : Function.Embedding α γ
              ⊢ Eq (fun b => Disjoint (Set.range ⇑a) (Set.range ⇑b)) fun f => ∀ (a_1 : β), M …
            -/
            ext f
            /-
              case h.a
              α : Type u_1
              β : Type u_2
              γ : Type u_3
              a : Function.Embedding α γ
              f : Function.Embedding β γ
              ⊢ Iff (Disjoint (Set.range ⇑a) (Set.range ⇑f)) (∀ (a_1 : β), Membership.mem (H …
            -/
            rw [← Set.range_subset_iff, Set.subset_compl_iff_disjoint_right, disjoint_comm]).trans
            /-
              🎉 no goals
            -/
        (codRestrict _ _)


/-- A combination of the above results, allowing us to turn one embedding over a sum type
into two dependent embeddings, the second of which avoids any members of the range
of the first. This is helpful for constructing larger embeddings out of smaller ones. -/
def sumEmbeddingEquivSigmaEmbeddingRestricted {α β γ : Type*} :
    (α ⊕ β ↪ γ) ≃ Σf : α ↪ γ, β ↪ ↥(Set.range f)ᶜ :=
  Equiv.trans sumEmbeddingEquivProdEmbeddingDisjoint
    prodEmbeddingDisjointEquivSigmaEmbeddingRestricted


/-- Embeddings from a single-member type are equivalent to members of the target type. -/
def uniqueEmbeddingEquivResult {α β : Type*} [Unique α] :
    (α ↪ β) ≃ β where
  toFun f := f default
  invFun x := ⟨fun _ => x, fun _ _ _ => Subsingleton.elim _ _⟩
  left_inv _ := by
    /-
      α : Type u_1
      β : Type u_2
      inst✝ : Unique α
      x✝ : Function.Embedding α β
      ⊢ Eq ((fun x => { toFun := fun x_1 => x, inj' := ⋯ }) ((fun f => f Inhabited.d …
    -/
    ext x
    /-
      case h
      α : Type u_1
      β : Type u_2
      inst✝ : Unique α
      x✝ : Function.Embedding α β
      x : α
      ⊢ Eq (((fun x => { toFun := fun x_1 => x, inj' := ⋯ }) ((fun f => f Inhabited. …
    -/
    simp_rw [Function.Embedding.coeFn_mk]
    /-
      case h
      α : Type u_1
      β : Type u_2
      inst✝ : Unique α
      x✝ : Function.Embedding α β
      x : α
      ⊢ Eq (x✝ Inhabited.default) (x✝ x)
    -/
    congr 1
    /-
      case h.h.e_6.h
      α : Type u_1
      β : Type u_2
      inst✝ : Unique α
      x✝ : Function.Embedding α β
      x : α
      ⊢ Eq Inhabited.default x
    -/
    exact Subsingleton.elim _ x
    /-
      🎉 no goals
    -/
                    /-
                      α : Type u_1
                      β : Type u_2
                      inst✝ : Unique α
                      x✝ : β
                      ⊢ Eq ((fun f => f Inhabited.default) ((fun x => { toFun := fun x_1 => x, inj'  …
                    -/
  right_inv _ := by simp
                    /-
                      🎉 no goals
                    -/


