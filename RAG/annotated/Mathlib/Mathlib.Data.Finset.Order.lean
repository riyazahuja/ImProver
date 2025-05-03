theorem Directed.finset_le {r : α → α → Prop} [IsTrans α r] {ι} [hι : Nonempty ι] {f : ι → α}
    (D : Directed r f) (s : Finset ι) : ∃ z, ∀ i ∈ s, r (f i) (f z) :=
  show ∃ z, ∀ i ∈ s.1, r (f i) (f z) from
                                                             /-
                                                               α : Type u
                                                               r : α → α → Prop
                                                               inst✝ : IsTrans α r
                                                               ι : Type u_1
                                                               hι : Nonempty ι
                                                               f : ι → α
                                                               D : Directed r f
                                                               s : Finset ι
                                                               z x✝ : ι
                                                               ⊢ Membership.mem 0 x✝ → r (f x✝) (f z)
                                                             -/
    Multiset.induction_on s.1 (let ⟨z⟩ := hι; ⟨z, fun _ ↦ by simp⟩)
                                                             /-
                                                               🎉 no goals
                                                             -/
      fun i _ ⟨j, H⟩ ↦
      let ⟨k, h₁, h₂⟩ := D i j
      ⟨k, fun _ h ↦ (Multiset.mem_cons.1 h).casesOn (fun h ↦ h.symm ▸ h₁)
        fun h ↦ _root_.trans (H _ h) h₂⟩


theorem Finset.exists_le [Nonempty α] [Preorder α] [IsDirected α (· ≤ ·)] (s : Finset α) :
    ∃ M, ∀ i ∈ s, i ≤ M :=
  directed_id.finset_le _

